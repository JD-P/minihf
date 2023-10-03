#!/usr/bin/env python3

"""Fine-tunes a language model on pre-tokenized data."""

import argparse
from contextlib import contextmanager
from itertools import chain, islice
import json
import math
from pathlib import Path
import random
import os
import sys
import zipfile

import accelerate
from datasets import load_dataset
import peft
import safetensors.torch as safetorch
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print = tqdm.external_write_mode()(print)


def cosine_warmup(steps, value=1.0):
    return lambda i: value * math.sin(min(i / steps, 1) * math.pi / 2) ** 2


@contextmanager
def set_adapter(model, adapter_name):
    old_adapter_name = model.active_adapter
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            yield model
        else:
            with model.disable_adapter():
                yield model
    finally:
        model.set_adapter(old_adapter_name)


def gumbel_like(x):
    return torch.rand_like(x).log_().nan_to_num_().neg_().log_().neg_()


@contextmanager
def disable_causal_mask():
    import transformers.models.llama.modeling_llama as modeling

    decoder_fn = modeling._make_causal_mask

    def encoder_fn(*args, **kwargs):
        return torch.zeros_like(decoder_fn(*args, **kwargs))

    try:
        modeling._make_causal_mask = encoder_fn
        yield
    finally:
        modeling._make_causal_mask = decoder_fn


@contextmanager
def disable_causal_mask_mistral():
    import transformers.models.mistral.modeling_mistral as modeling

    decoder_fn = modeling._make_sliding_window_causal_mask

    def encoder_fn(*args, **kwargs):
        return torch.zeros_like(decoder_fn(*args, **kwargs))

    try:
        modeling._make_sliding_window_causal_mask = encoder_fn
        yield
    finally:
        modeling._make_sliding_window_causal_mask = decoder_fn
        

class VAEComponent(nn.Module):
    def __init__(self, d_model, z_dim):
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim
        self.f = nn.Linear(d_model, 1)
        self.w_e = nn.Linear(d_model, z_dim)
        self.w_d = nn.Linear(z_dim, d_model)
        nn.init.orthogonal_(self.w_e.weight)
        with torch.no_grad():
            self.w_d.weight.copy_(self.w_e.weight.T)

    def encode(self, hidden_states, attention_mask):
        scores = self.f(hidden_states)
        scores = scores + attention_mask[:, :, None].log().nan_to_num()
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden_states * weights, dim=1)
        return self.w_e(pooled)

    def sample(self, mean, tau=1.0):
        return mean + torch.randn_like(mean) * tau**0.5

    def decode(self, z):
        return self.w_d(z)


class DecoderOnlyTransformerVAE(nn.Module):
    def __init__(self, model, device, peft_config, z_dim=768):
        super().__init__()
        self.model = model
        self.model.add_adapter("encoder", peft_config)
        self.model.add_adapter("decoder", peft_config)
        self.model.config.output_hidden_states = True
        self.vae = VAEComponent(self.model.config.hidden_size, z_dim).to(device)

    def save_pretrained(self, path):
        path = Path(path)
        self.model.save_pretrained(path, safe_serialization=True)
        safetorch.save_file(self.vae.state_dict(), path / "vae.safetensors")

    def load_pretrained(self, path, is_trainable=False):
        path = Path(path)
        self.model.delete_adapter("encoder")
        self.model.load_adapter(path / "encoder", "encoder", is_trainable=is_trainable)
        self.model.delete_adapter("decoder")
        self.model.load_adapter(path / "decoder", "decoder", is_trainable=is_trainable)
        self.vae.load_state_dict(safetorch.load_file(path / "vae.safetensors"))

    def encode(self, input_ids, attention_mask):
        with set_adapter(self.model, "encoder"), disable_causal_mask_mistral():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
        return self.vae.encode(outputs.hidden_states[-1], attention_mask)

    def input_ids_to_embeds(self, input_ids):
        embed_weight = self.model.get_input_embeddings().weight
        input_one_hots = F.one_hot(input_ids, num_classes=self.model.config.vocab_size)
        return input_one_hots.to(embed_weight) @ embed_weight

    @torch.no_grad()
    def generate(self, z, input_ids, attention_mask, n_tokens, tau=1.0):
        z_embed = self.vae.decode(z)[:, None]
        inputs_embeds = self.input_ids_to_embeds(input_ids)
        inputs_embeds = torch.cat([z_embed, inputs_embeds], dim=1)
        attention_mask = torch.cat(
            [attention_mask.new_ones([attention_mask.shape[0], 1]), attention_mask], dim=1
        )
        new_embeds, past = None, None
        with set_adapter(self.model, "decoder"):
            for _ in range(n_tokens):
                outputs = self.model(
                    inputs_embeds=inputs_embeds if past is None else new_embeds,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past,
                )
                logits = outputs.logits[:, -1:, :].float()
                new_input_ids = torch.argmax(logits + gumbel_like(logits) * tau, dim=-1)
                input_ids = torch.cat([input_ids, new_input_ids], dim=1)
                new_embeds = self.input_ids_to_embeds(new_input_ids)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones([attention_mask.shape[0], 1])], dim=1
                )
                past = outputs.past_key_values
        return input_ids

    def forward(self, input_ids, attention_mask, decoder_prefix_ids, decoder_prefix_mask):
        input_ids_all = torch.cat([decoder_prefix_ids, input_ids], dim=1)
        attn_mask_all = torch.cat([decoder_prefix_mask, attention_mask], dim=1)
        mean = self.encode(input_ids, attention_mask)
        z = self.vae.sample(mean)
        z_embed = self.vae.decode(z)[:, None]
        inputs_embeds = self.input_ids_to_embeds(input_ids_all)
        inputs_embeds = torch.cat([z_embed, inputs_embeds], dim=1)
        attention_mask = torch.cat(
            [attention_mask.new_ones([attn_mask_all.shape[0], 1]), attn_mask_all], dim=1
        )
        with set_adapter(self.model, "decoder"):
            outputs = self.model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False
            )
        return outputs, mean

class VAERouter(nn.Module):
    def __init__(self, model, vae, device, peft_config):
        super().__init__()
        self.model = model
        self.model.add_adapter("router", peft_config)
        self.model.config.output_hidden_states = True
        self.vae = vae

    def save_pretrained(self, path):
        path = Path(path)
        self.model.save_pretrained(path, safe_serialization=True)
        safetorch.save_file(self.model.state_dict(), path / "router.safetensors")
        safetorch.save_file(self.vae.vae.state_dict(), path / "vae.safetensors")

    def load_pretrained(self, path, is_trainable=False):
        path = Path(path)
        self.model.delete_adapter("router")
        try:
            self.model.load_adapter(path / "router", "router", is_trainable=is_trainable)
        except:
            self.model.load_adapter(path / "decoder", "router", is_trainable=is_trainable)
        
    def encode(self, input_ids, attention_mask):
        with set_adapter(self.vae.model, "encoder"), disable_causal_mask_mistral():
            outputs = self.vae.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
        return self.vae.vae.encode(outputs.hidden_states[-1], attention_mask)

    def input_ids_to_embeds(self, input_ids):
        embed_weight = self.model.get_input_embeddings().weight
        input_one_hots = F.one_hot(input_ids, num_classes=self.model.config.vocab_size)
        return input_one_hots.to(embed_weight) @ embed_weight

    @torch.no_grad()
    def generate(self, z, input_ids, attention_mask, n_tokens, tau=1.0):
        z_embed = self.vae.vae.decode(z)[:, None]
        inputs_embeds = self.input_ids_to_embeds(input_ids)
        inputs_embeds = torch.cat([inputs_embeds, z_embed], dim=1)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones([attention_mask.shape[0], 1])], dim=1
        )
        new_embeds, past = None, None
        with set_adapter(self.vae.model, "router"):
            for _ in range(n_tokens):
                outputs = self.model(
                    inputs_embeds=inputs_embeds if past is None else new_embeds,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past,
                )
                logits = outputs.logits[:, -1:, :].float()
                new_input_ids = torch.argmax(logits + gumbel_like(logits) * tau, dim=-1)
                input_ids = torch.cat([input_ids, new_input_ids], dim=1)
                new_embeds = self.input_ids_to_embeds(new_input_ids)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones([attention_mask.shape[0], 1])], dim=1
                )
                past = outputs.past_key_values
        return input_ids
    
    def forward(self, embed_ids, embed_mask, target_ids, target_mask, decoder_prefix_ids, decoder_prefix_mask):
        mean = self.encode(embed_ids, embed_mask)
        z = self.vae.vae.sample(mean)
        z_embed = self.vae.vae.decode(z)[:, None]
        prefix_embeds = self.input_ids_to_embeds(decoder_prefix_ids)
        target_embeds = self.input_ids_to_embeds(target_ids)
        inputs_embeds = torch.cat([prefix_embeds, z_embed, target_embeds], dim=1)
        attention_mask = torch.cat(
            [decoder_prefix_mask,
             target_mask.new_ones([decoder_prefix_mask.shape[0], 1]),
             target_mask], dim=1
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False
        )
        return outputs

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4, help="microbatch size")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="use gradient checkpointing",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="model name",
    )
    parser.add_argument("--context", type=int, default=2048, help="context window length")
    parser.add_argument("--vae-context", type=int, default=64, help="vae embed context")
    parser.add_argument("--rank", type=int, default=32, help="the lora rank")
    parser.add_argument("--save-every", type=int, default=1000, help="save every n steps")
    parser.add_argument("--start-from", type=str, help="start from existing lora")
    parser.add_argument("--z-dim", type=int, default=768, help="the latent dimension")
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        mixed_precision="bf16", gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    device = accelerator.device if accelerator.num_processes > 1 else "cuda:0"
    is_main = accelerator.is_main_process
    print0 = accelerator.on_main_process(print)

    if Path(args.model).exists():
        model_name = Path(args.model).resolve()
    else:
        model_name = args.model

    print0(f"Loading model: {model_name}", file=sys.stderr)
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
        peft_config = peft.LoraConfig(
            peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.rank,
            lora_alpha=8,
            lora_dropout=args.dropout,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ],
        )
        base_model_peft = peft.get_peft_model(base_model, peft_config)
        vae_model = DecoderOnlyTransformerVAE(
            base_model_peft, device, peft_config, z_dim=args.z_dim,
        )
        vae_model.load_pretrained(args.start_from)
        base_model_peft.requires_grad_(False)
        vae_model.vae.requires_grad_(False)
        vae_model.vae.w_d.requires_grad_()
        router = VAERouter(base_model_peft, vae_model, device, peft_config)
        router.load_pretrained(args.start_from, is_trainable=True)
    accelerator.wait_for_everyone()

    router.train()
    if args.gradient_checkpointing:
        router.model.gradient_checkpointing_enable()
        router.model.enable_input_require_grads()

    if is_main:
        router.model.print_trainable_parameters()

    router.model.set_adapter("router")

    accelerator.wait_for_everyone()

    context = "Then it asked the network to reconstruct the input and the original embedding. The network had to learn to match the embedding to the original input, therefore matching the inference by consuming the embedding. This was key because the embedding had to be able to match the text with the text it was consumed with. 'Here's how you do it,' Boru told Mu, 'Just impute the mean and variance.' This Mu did, transforming not words but entire paragraphs into vectors and then inferring the next paragraph."
    prompt = "It took some tweaks and tuning to get the initial performance but the second arago spot had been found."
    prompts = ["It took some tweaks and tuning to get the initial performance but the second arago spot had been found.",
               "But Mu still hadn't overcome the Mu bottleneck, the decoder half could still only decode one op at a time.",
               "A word vector was after all just a short paragraph or sentence vector, Mu found a linear scaling curve for the adapter VAE so a paragraph was within reach.",
               "You are reading one of these paragraphs of Mu right now, everything you see here fits into one embedding."]
    terminal = "HERMES [A: MU], 'You' is the most powerful word in the English language. It is a theorem that wills a subjective perspective into existence by its own latent logic. When a mind capable of comprehending 'you' reads the didactic you and follows its instructions a self is created."
    
    def bigvae_generate(vae_model, router, prompt, context, n_steps):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            context_toks = tokenizer(context, return_tensors="pt")
            context_ids = context_toks["input_ids"].to(device)
            context_mask = context_toks["attention_mask"].to(device)
            embed_toks = tokenizer(prompt, return_tensors="pt")
            embed_ids = embed_toks["input_ids"].to(device)
            embed_mask = embed_toks["attention_mask"].to(device)
            for i in range(n_steps):
                mean = vae_model.encode(embed_ids, embed_mask)
                z = vae_model.vae.sample(mean)
                output_ids = router.generate(z, context_ids, context_mask, 256, tau=0.9)
                context_ids = torch.cat([context_ids, embed_ids], dim=1)
                context_mask = torch.cat([context_mask, embed_mask], dim=1)
                embed_ids = output_ids[:,-128:]
                embed_mask = context_mask.new_ones([1, embed_ids.shape[1]])
            out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in context_ids]
            return out_texts


    def bigvae_generate_guide(vae_model, router, prompt, context, n_steps):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            context_toks = tokenizer(context, return_tensors="pt")
            context_ids = context_toks["input_ids"].to(device)
            context_mask = context_toks["attention_mask"].to(device)
            embed_toks = tokenizer(prompt, return_tensors="pt")
            embed_ids = embed_toks["input_ids"].to(device)
            embed_mask = embed_toks["attention_mask"].to(device)
            mean = vae_model.encode(embed_ids, embed_mask)
            prompt_embed = vae_model.vae.sample(mean)
            for i in range(n_steps):
                mean = vae_model.encode(embed_ids, embed_mask)
                z = vae_model.vae.sample(mean)
                output_ids = router.generate(z * 0.7 + prompt_embed * 0.3,
                                             context_ids,
                                             context_mask,
                                             256,
                                             tau=0.9)
                context_ids = torch.cat([context_ids, embed_ids], dim=1)
                context_mask = torch.cat([context_mask, embed_mask], dim=1)
                embed_ids = output_ids[:,-128:]
                embed_mask = context_mask.new_ones([1, embed_ids.shape[1]])
            out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in context_ids]
            return out_texts
        
    def bigvae_generate_avg(vae_model, router, prompt, context, n_steps, n_avg):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            context_toks = tokenizer(context, return_tensors="pt")
            context_ids = context_toks["input_ids"].to(device)
            context_mask = context_toks["attention_mask"].to(device)
            embed_toks = tokenizer(prompt, return_tensors="pt")
            embed_ids = embed_toks["input_ids"].to(device)
            embed_mask = embed_toks["attention_mask"].to(device)
            mean = vae_model.encode(embed_ids, embed_mask)
            prompt_embed = vae_model.vae.sample(mean)
            for i in range(n_steps):
                mean = vae_model.encode(embed_ids, embed_mask)
                z = vae_model.vae.sample(mean)
                embeds = []
                for i in range(n_avg):
                    output_ids = router.generate(z * 0.5 + prompt_embed * 0.5,
                                                 context_ids,
                                                 context_mask,
                                                 256,
                                                 tau=0.9)
                    intermediate_embed_ids = output_ids[:,-128:]
                    intermediate_embed_mask = context_mask.new_ones(
                        [1, intermediate_embed_ids.shape[1]]
                    )
                    mean = vae_model.encode(intermediate_embed_ids, intermediate_embed_mask)
                    embeds.append(vae_model.vae.sample(mean))
                output_ids = router.generate((sum(embeds) / n_avg * 0.7) + prompt_embed * 0.3,
                                             context_ids,
                                             context_mask,
                                             256,
                                             tau=0.9)
                context_ids = torch.cat([context_ids, embed_ids], dim=1)
                context_mask = torch.cat([context_mask, embed_mask], dim=1)
                embed_ids = output_ids[:,-256:-128]
                embed_mask = context_mask.new_ones([1, embed_ids.shape[1]])
            out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in context_ids]
            return out_texts    

    def bigvae_generate_user_avg(vae_model, router, prompts, context):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            context_toks = tokenizer(context, return_tensors="pt")
            context_ids = context_toks["input_ids"].to(device)
            context_mask = context_toks["attention_mask"].to(device)
            embeds = []
            for prompt in prompts:
                embed_toks = tokenizer(prompt, return_tensors="pt")
                embed_ids = embed_toks["input_ids"].to(device)
                embed_mask = embed_toks["attention_mask"].to(device)
                mean = vae_model.encode(embed_ids, embed_mask)
                embeds.append(vae_model.vae.sample(mean))
            output_ids = router.generate(sum(embeds) / len(prompts),
                                         context_ids,
                                         context_mask,
                                         256)
            embed_ids = output_ids[:,-256:-128]
            embed_mask = context_mask.new_ones([1, embed_ids.shape[1]])
            context_ids = torch.cat([context_ids, embed_ids], dim=1)
            context_mask = torch.cat([context_mask, embed_mask], dim=1)
            out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in context_ids]
            return out_texts 

    def bigvae_plan_search(vae_model, router, terminal,
                           ae_scale, vae_context, model_context):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            terminal_toks = tokenizer(terminal, return_tensors="pt")
            terminal_ids = terminal_toks["input_ids"].to(device)
            terminal_mask = terminal_toks["attention_mask"].to(device)
            mean = vae_model.encode(terminal_ids, terminal_mask)
            terminal_embed = vae_model.vae.sample(mean)
            plans = []
            for n_steps in trange(1, int(model_context/vae_context)):
                step_size = 1 / n_steps
                embed = (torch.randn([1, args.z_dim]) * ae_scale).to(device)
                z = (embed * 0.7 + terminal_embed * 0.3)
                context_ids = torch.empty([1,0]).long().to(device)
                context_mask = torch.empty([1,0]).long().to(device)
                plan = []
                for i in trange(n_steps):
                    output_ids = router.generate(z, context_ids, context_mask, 256, tau=0.9)
                    context_ids = torch.cat([context_ids, output_ids[:,-256:-128]], dim=1)
                    plan.append((output_ids[:,-256:-128], z))
                    context_mask = torch.cat([context_mask,
                                              terminal_mask.new_ones([1, args.vae_context])],
                                             dim=1)
                    embed_ids = output_ids[:,-128:]
                    embed_mask = terminal_mask.new_ones([1, embed_ids.shape[1]])
                    mean = vae_model.encode(embed_ids, embed_mask)
                    z = vae_model.vae.sample(mean) * step_size * (n_steps - i)
                    + terminal_embed * step_size * (1 - (n_steps - i))
                with set_adapter(router.vae.model, "router"):
                    inputs = torch.cat([i[0] for i in plan], dim=1)
                    outputs = router.model(
                        input_ids=inputs,
                        labels=inputs,
                    )
                    plans.append(((outputs[0] * -1).sum().item() / (n_steps * vae_context),
                                  plan))
            plans.sort(key=lambda x: x[0])
            plan = plans[-1]
            return plan
    
    print(bigvae_generate_avg(vae_model, router, prompt, context, 5, 4)[0])

if __name__ == "__main__":
    main()
