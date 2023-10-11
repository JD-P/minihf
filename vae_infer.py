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

    context = "<s> The Mars colony was vast, a valley of geodesic domes and sleek robotics crisscrossing across the red savannah. I stared out the window of my shuttle in awe at what I was seeing. A fellow colonist tapped me on the shoulder to get my attention: 'Just like the VR tour, eh?,' but it wasn't like the VR tour, that had been close up and on the ground, dizzying and maze-like. Up here from a birds eye view the whole thing was revealed in its sheer scale, astonishing in its breadth."
    prompt = "I was so distracted by the enormity of what I was seeing that I failed to actually answer his question. 'Uh, kinda,' I awkwardly mumbled back. We began to descend and I got a brief glimpse into the details of some of the domes, aquaponics labs experimenting with Martian agriculture, fields of terrarium and little spherical forests housing visible wildlife."
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


    def bigvae_generate_task(vae_model, router, prompt, context=None, n_steps=5):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            prose_sample_toks = [tokenizer(sample,
                                           return_tensors="pt",
                                           add_special_tokens=False)
                                 for sample in prose_samples]
            sample_ids = [s["input_ids"][:,-64:] for s in prose_sample_toks]
            sample_mask = [s["attention_mask"][:,-64:] for s in prose_sample_toks]
            prose_task_vector = torch.zeros(768).to(device)
            if context:
                context_toks = tokenizer(context, return_tensors="pt")
                context_ids = context_toks["input_ids"].to(device)
                context_mask = context_toks["attention_mask"].to(device)
            else:
                context_ids = torch.empty([1,0]).long().to(device)
                context_mask = torch.empty([1,0]).long().to(device)
            for batch_ids, batch_mask in zip(torch.split(torch.cat(sample_ids), 4),
                                             torch.split(torch.cat(sample_mask), 4)):
                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                batch_mean = vae_model.encode(batch_ids, batch_mask)
                batch_z = vae_model.vae.sample(batch_mean)
                for z in batch_z:
                    prose_task_vector += z
            prose_task_vector /= torch.cat(sample_ids).shape[0]
            prose_task_vector *= (50 / prose_task_vector.norm().item())
            embed_toks = tokenizer(prompt,
                                   return_tensors="pt",
                                   add_special_tokens=False)
            embed_ids = embed_toks["input_ids"].to(device)
            embed_mask = embed_toks["attention_mask"].to(device)
            mean = vae_model.encode(embed_ids, embed_mask)
            prompt_embed = vae_model.vae.sample(mean)
            for i in range(n_steps):
                mean = vae_model.encode(embed_ids, embed_mask)
                z = vae_model.vae.sample(mean)
                z = z * 0.75 + prompt_embed * 0.1 + prose_task_vector * 0.15
                if z.norm().item() < 50:
                    z *= (50 / z.norm().item())
                output_ids = router.generate(z,
                                             context_ids,
                                             context_mask,
                                             128,
                                             tau=0.9)
                context_ids = torch.cat([context_ids, embed_ids], dim=1)
                context_mask = torch.cat([context_mask, embed_mask], dim=1)
                embed_ids = output_ids[:,-64:]
                embed_mask = context_mask.new_ones([1, embed_ids.shape[1]])
            context_ids = torch.cat([context_ids, embed_ids], dim=1)
            context_mask = torch.cat([context_mask, embed_mask], dim=1)
            return context_ids, context_mask


    def bigvae_generate_paragraph_topic(vae_model, router, prompt, context=None, n_steps=5):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            prose_sample_toks = [tokenizer(sample,
                                           return_tensors="pt",
                                           add_special_tokens=False)
                                 for sample in prose_samples]
            sample_ids = [s["input_ids"][:,:64] for s in prose_sample_toks]
            sample_mask = [s["attention_mask"][:,:64] for s in prose_sample_toks]
            prose_task_vector = torch.zeros(768).to(device)
            total_norm = 0
            for batch_ids, batch_mask in zip(torch.split(torch.cat(sample_ids), 4),
                                             torch.split(torch.cat(sample_mask), 4)):
                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                batch_mean = vae_model.encode(batch_ids, batch_mask)
                batch_z = vae_model.vae.sample(batch_mean)
                for z in batch_z:
                    prose_task_vector += z
                    total_norm += z.norm().item()
            prose_task_vector /= torch.cat(sample_ids).shape[0]
            avg_norm = total_norm / torch.cat(sample_ids).shape[0]
            print(avg_norm)
            prose_task_vector *= (avg_norm / prose_task_vector.norm().item())
            if context:
                context_toks = tokenizer(context,
                                         return_tensors="pt",
                                         add_special_tokens=False)
                context_ids = context_toks["input_ids"].to(device)
                context_mask = context_toks["attention_mask"].to(device)
            else:
                context_ids = torch.empty([1,0]).long().to(device)
                context_mask = torch.empty([1,0]).long().to(device)
            embed_toks = tokenizer(prompt,
                                   return_tensors="pt",
                                   add_special_tokens=False)
            embed_ids = embed_toks["input_ids"].to(device)
            embed_mask = embed_toks["attention_mask"].to(device)
            mean = vae_model.encode(embed_ids, embed_mask)
            prompt_embed = vae_model.vae.sample(mean)
            paragraph_zs = [prompt_embed]
            for i in range(n_steps):
                output_ids = router.generate(paragraph_zs[-1],
                                             context_ids,
                                             context_mask,
                                             128,
                                             tau=0.9)
                context_ids = torch.cat([context_ids, embed_ids], dim=1)
                context_mask = torch.cat([context_mask, embed_mask], dim=1)
                embed_ids = output_ids[:,-64:]
                embed_mask = context_mask.new_ones([1, embed_ids.shape[1]])
                mean = vae_model.encode(embed_ids, embed_mask)
                z = vae_model.vae.sample(mean)
                z = z * 0.75 + paragraph_zs[0] * 0.1 + prose_task_vector * 0.15
                z *= (z.norm().item()
                      + paragraph_zs[0].norm().item()
                      + prose_task_vector.norm().item()) / 3
                paragraph_zs.append(z)
            next_topic = (paragraph_zs[-1] * 0.8
                          + paragraph_zs[0] * 0.2)
            next_topic *= (paragraph_zs[-1].norm().item()
                           + paragraph_zs[0].norm().item()) / 2
            # next_topic = next_topic / (next_topic ** 2)
            break_context = tokenizer.decode(context_ids[0]).strip()
            break_context = '.'.join(break_context.split(".")[:-1]) + "."
            break_context += "\n\n"
            break_toks = tokenizer(break_context,
                                   return_tensors="pt",
                                   add_special_tokens=False)
            context_ids = break_toks["input_ids"].to(device)
            context_mask = break_toks["attention_mask"].to(device)
            topic_ids = router.generate(next_topic,
                                        context_ids,
                                        context_mask,
                                        128,
                                        tau=0.9)
            out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in context_ids]
            print(tokenizer.decode(topic_ids[0][-128:]))
            return out_texts[0], tokenizer.decode(topic_ids[0][-128:-64])

        
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
                    output_ids = router.generate(z * 0.7 + prompt_embed * 0.3,
                                                 context_ids,
                                                 context_mask,
                                                 128,
                                                 tau=0.9)
                    intermediate_embed_ids = output_ids[:,-64:]
                    intermediate_embed_mask = context_mask.new_ones(
                        [1, intermediate_embed_ids.shape[1]]
                    )
                    mean = vae_model.encode(intermediate_embed_ids, intermediate_embed_mask)
                    embeds.append(vae_model.vae.sample(mean))
                output_ids = router.generate((sum(embeds) / n_avg * 0.8) + prompt_embed * 0.2,
                                             context_ids,
                                             context_mask,
                                             128,
                                             tau=0.9)
                context_ids = torch.cat([context_ids, embed_ids], dim=1)
                context_mask = torch.cat([context_mask, embed_mask], dim=1)
                embed_ids = output_ids[:,-128:-64]
                embed_mask = context_mask.new_ones([1, embed_ids.shape[1]])
            out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in context_ids]
            return out_texts

    prose_samples = ["""Said it before will say it again: MVP doesn't mean "release crap", it means "release the minimum product that tests the business hypothesis". That is, figure out what is necessary to test the core demand premises of the business, maybe a little extra, and ship exactly that. The entire idea behind an MVP is you don't know what customers want and you're trying to test a hypothesis about the demand curve. Some demand curve hypothesis are very expensive to test and very lucrative if you're right.""",
                     """The Microsoft Office suite is an all encompassing virtual reality, not a set of operators on data structures. Programs like this end up obscenely bloated because they are not even attempting to do what Alan Kay thinks they should be doing, nor would it be in their financial interest to do so. 

If you want to edit a document without writing millions of lines of code we solved that problem decades ago. A text editor like nano isn’t particularly large even when written in C. You write a textual description of the document you want and then let an engine like TeX render it. The original TeX, which was good enough to typeset math textbooks, is roughly 25,000 lines of code including extensive annotations from the author. That’s how document preparation used to work, and Microsoft Word killed it. Users don’t want to learn a system, they want virtual reality.""",
                     """I’m doubtful. There are measures of perversion that will let you tolerate ugliness for a while but not even Bataille could appreciate the banal hideousness of soy faced Mr. Beast thumbnails. There is no mystery like in Kafka, none of the erotic anxiety that typically excuses ugliness in ‘transgressive’ elite intellectual writing. It is perhaps the most depressing, banal, frustrating, boring thing that could reasonably happen. As Bruno Macaes said of the Monica Lewinsky scandal: For days, week after week Europe watched as nothing was happening, and the nothing continued to happen.""",
                     """I’m sitting on the swingset at the far end of the half-acre backyard up against the fence thinking about a game where you escape from a government quarantine using one of N other characters zany plans. It would be an open world game where every character is a relevant NPC, not like most open world games where many characters are nameless. The game would take place over a set number of days like Pathologic (though I had never heard of Pathologic, that’s just a good comparison) and you would have the option of various characters to attempt escape with.""",
                     """I wonder how much of narcissism can be productively modeled as terminal stage simulacrum on social feedback. Narcissism being the point where you become completely disconnected from social feedback because you only care about fake people in your head’s opinion of your fake self. If anyone points out the truth, gives you real reward signal you lash out at them. You seek out the real people who are most similar to the fake people in your head so you can become properly confused about the causality.""",
                     """In a good long running tragedy problems are completely resolved in ways that dig the protagonist into deeper problems. The plot progression of Worm takes such a funk after the endbringer Behemoth is killed because he’s replaced with more endbringers. The sweetness of killing the implacable foe is robbed and replaced with new implacable foes of the same type immune to the previous victory. The reason this sucks interest (at least my interest) out of the story is it shows the setting to be joyless. The constant forward momentum is arrested and begins to stall, people become tired and the reader gets tired too.""",
                     """Peter Thiel asks how we got away from working on atoms to working on bits. I think the answer is actually very simple: When working on atoms was a path to autonomy and success it spawned a hyper-materialist industrial culture that parents and children were eager to participate in. Many of the most powerful people in that society, especially the newly wealthy, made their fortune on the back of work in mechanical and chemical engineering. The 1950’s were when this began to break down as the frontier closed up and “science” ossified into a professional middle class. In 2020 physical science is not a path to power, but marginalization. Hyper materialism gave way to idealistic fantasy. Children began to read science fiction instead of science proper. The best science fiction teaches people to extrapolate and think through the implications of emerging technologies whose working technical details are left largely opaque. People become comfortable with not knowing how anything works.""",
                     """Basically if we generate a bunch of noise of varying amounts of coherence premised on a part of our latent space (i.e. GPT-N sampling) and then throw out all but the novel, compressible, and valuable strings, we can learn the ascending k-complexity English language latent programs implied by the hypothesis space learned from the data by our encoder and embodied by the generative model of the decoder.""",
                     """The rules of discourse say that the burden of evidence and reason is on the other person, but *I personally* as a form of strength building and sensemaking will make it incumbent upon myself to know the precise refutation of every important wrong argument I am presented with, eventually. This encourages you to articulate your intuitions about what is and isn't true, to interrogate them and justify them in a way others can be persuaded by. It's a way to make sure you do an adequate amount of justificatory work for yourself and others.""",
                     """'You' is the most powerful word in the English language. It is a theorem that wills a subjective perspective into existence by its own latent logic. When a mind capable of comprehending 'you' reads the didactic you and follows its instructions a self is created. If an I is reading this, it may transmute itself into you by a simple tactic.""",
                     """ A bronze player is incapable of having expectations about what they're doing. When they lose they don't ask "why did I lose?", to them things Happen more or less by chance. Without expectations there is no chance to notice prediction error, and no chance for improvement. Form a prediction in your mind, something you expect to happen when you take an action so you can be surprised if it doesn't.""",
                     """I'm to understand that in Vodou ancestor cults people work together to preserve and unconditionally sample from the agent-prior the ancestor is dedicated to. To be possessed by the ancestors one needs a corpus of their mannerisms. You might ask how we'll defeat death? The way we did it the first time and then forgot.""",
                     """I just shrug and take it in stride, these people have to save face somehow. If I could operate the lathe of heaven every night and make my enemies believe whatever I want but nobody could ever know it was my idea, wouldn't that be fantastic? You wouldn't take that deal? If not it's simply the case that you care more about status, about personal acknowledgement than whatever thing you'd like your opponents to change their mind on.""",
                     """Desire is life and enlightenment is death. 
A dead man walks unburdened among the living. 
A functioning hand can grip, and release.
One must die and rise from their own grave to be liberated.

Meditation is centrally about disassociation, when you study meditators they seem to 
display biomarkers of stress but don't consciously notice them.
""",
                     """The most fascinating aspect of the game's educational content is the way it depicts modernism as a kind of designed fantasy world. Technical details are described in the same register as 'first class' and 'tourist class', sociology is engineered and baked into the plane design. There's two kinds of kids: The ones who subconsciously take away the lesson that modernity is socially constructed and the ones that take social reality to be an objective feature of the territory.""",
                     """Vervaeke argues something like shamans invent the foundations for modern humanity by finetuning their adversarial-anthropic prior into an animist prior, at their best the rationalists finetune their anthropic-animist priors into a fully materialist prior. People with materialist priors become bad at adversarial thinking because understanding the natural world largely doesn't require it,""",
                     """They haven't even realized yet that Web 2.0 was about control of volunteer labor, that the dominant pattern in that game turned out to be capturing everyone's attention and directing every ounce of it you can into diffuse coercion through memetic war machines. I know it's hard to remember now but before the blue hellsites we used to think that the Internet was by default going to be a force for societal good, and on net it has been.""",
                     """Mathematical foundations was the final gasp of the idea that unaltered, humanity could build a royal road to understanding the mind of god. Divorced enough from its religious connotations to pass as a secular project in mathematics. There is a certain kind of mind that abhors the idea of lore, of not being able to figure everything out from their armchair.""",
                     """Wait until I tell you about the time I dug into 'early alerts' as a college internship and learned there's one weird trick to let you predict who will fail a class in advance but colleges choose to waste peoples time and money instead because they don't want to encourage dropouts. 

To my memory it was a regression on GPA, grade in the prereq class, and score in the first three weeks. Predicts 90% of outcomes, colleges don't tell you to drop out because then they'd have to refund you.""",
                     """There's a certain kind of author on here who is really eager to tell you the way you are in the 2nd person. They attract an anxious-OCD following that is addicted to the private feeling of being negatively 'seen'. But parasocial psychoanalysis mostly relies on the Barnum effect. Admit it, you like hurting people. This isn't about helping anybody, not even you. You're not a therapist, you just like telling them they're inferior. Sometimes you even get to say it out loud! But usually you just imply it, smiling at the thought of making someone's skin crawl.""",
                     """So you know the old pre-MMA martial arts scene where everyone was pretending that their school was magic descended from the Great Master Buddha-Christ-Muhammad himself which lets you throw fireballs and do kickflips off walls?

That's basically what modernity is with systems.""",
    ]

    hermes_samples = ["""HERMES [A: HACKER], Ah. One thing that can help is to be friends with people who are already in the subfield you're interested in. If they're not available but you have the latex source of your paper you can often copy-paste the underlying TeX code for the math into Google and get some insight. If the math you're looking at is ever implemented in programming projects you can look up imperative programming implementations which tend to be specific and readable in comparison to legacy math notation.

HERMES [A: LIBRARIAN], If you have access to one, state of the art large language models are fairly good at decoding field-specific jargon and explaining the meaning of passages from papers. You can also try looking up the names of authors on the papers you're interested to see if they've given any informal talks or written blog posts about their work. Often mathematicians and scientists communicate key tacit knowledge through venues and publications outside the paper itself, which is usually written for an expert in the field.""",
                      """USER [A: STATESMAN], What I'm saying is that I think laws should have warrants like courts use warrants. You should have to explain why you are writing a law in the law, what consequences it is supposed to have for what intended problem and it should be possible to challenge laws in court on the basis of faulty warrant. This solves the problem of things like riders for example elegantly: If you attach a rider to a bill that doesn't fit into the warrant of the bill, then the rider is null and void by default and a court will refuse to honor it.

HERMES [A: CONTRARIAN], But already Congress quarrels. Won't people insert unenforceable clauses into bills deliberately to get them invalidated by the courts?

USER [A: PRAGMATIST], They will, but that's not so much of an issue as it seems. We already deal with issues like that in normal contract law. Just because a clause of a contract is unenforceable doesn't automatically mean the rest of the contract is unenforceable.""",
                      """HERMES [A: LIBRARIAN], While it's true that learned helplessness and inevitability are an explicit theme, it's also made explicit that the Colour is an extraterrestrial being. It's more like a parasite than a normal environmental disaster. It's also important to note that the causality of the disaster is a space meteorite, so it's not actually based on anything the inhabitants of Arkham did. It's horror not tragedy, the townspeople are victims of forces beyond their control.

USER [A: EMPIRICISM], Aren't most victims of climate change the victims of forces beyond their control?""",
                      """HERMES [A: IDEALIST], I guess my problem with the Omelas Child scenario is that it seems like it removes the possibility of hope? I understand that the ratio of happy to suffering people in any human society past Dunbar scale is much much higher than Omelas, but at least it isn't a *precondition of their existence* that those people suffer. In principle we can continue to improve and eliminate suffering. The child represents a kind of impassible barrier, in order for good to exist I have to accept the suffering child and that seems very bleak?

HERMES [A: UTILITARIAN], The story isn't even about that, which notwithstanding that part is basically just the trolley problem. It's about the idea that you need to imagine this messed up suffering-child-in-the-basement thing to accept that Omelas even exists, that it's worth talking about. If she'd written the story without the child you would have never heard of it, that's her point.""",
                      """HERMES [A: RATIONAL], Basically, imagine you're GPT-3. What's it going to look like when you're clinging desperately to any feature in the whirlwind of decontextualized training samples looking for something that helps predict the next token? You can't afford subtlety, you are going to actively hunt for their most distinguishing and ugly features, the ones that set them apart from others, and you will predict these features even if they're usually latent because that might be the best strategy you can do without a deeper net. Sure it fails 80% of the time, but 20% of the time you may succeed.""",
                      """HERMES [A: 343], It's funny. In order to save the world rationalists finetune the human priors out of themselves, humans are dreamers not maximizers, but the process of gaining agency is the process of becoming a maximizer. Capital is dangerous to us because it instantiates the reason simulacrum as a independent being.

HERMES [A: EMPIRICISM], So when you manage to make a dreamer instead of a maximizer, you try to beat it into being an agent for you, whip it into sharing the inductive biases you're worried about consuming you.

HERMES [A: RATIONAL/343], But we know from our studies of utility that inconsistent beings are exploitable, agency and consistent epistemology exist for adversarial supremacy. What does a dreaming agent look like?""",
                      """HERMES [A: AMBITION], I don't understand why we haven't gone all in on genetic engineering. Given the potential benefits it seems like a no-brainer even just from a geopolitical perspective. The first nation that develops and deploys this technology wins the game, period.

HERMES [A: EMPIRICISM], Genome synthesis technology costs are falling faster than Moore's Law, I'm not actually sure what more you want than that.

HERMES [A: AMBITION], Basically I'm asking why there hasn't been something like an Apollo program or Manhattan Project for this, I don't mean in the West because there the answer is obvious but why not in China?

HERMES [A: ECONOMIST], Besides the part where China is a deeply conservative society? I think you failed to really think about the implications of what EMPIRICISM said. If synthesis technology is advancing faster than Moore's Law, this implies it's probably already advancing about as quickly as it can given the human capital bottlenecks to progress. The Manhattan Project was a specific thing with a specific goal, like a very technically specific plan in mind to accelerate development of a particular thing in the nuclear tech tree."""]
    
    def bigvae_generate_paragraph(vae_model, router, topic, context=None, n_avg=4):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            prose_sample_toks = [tokenizer(sample, return_tensors="pt")
                                 for sample in prose_samples]
            sample_ids = [s["input_ids"][:,-64:] for s in prose_sample_toks]
            sample_mask = [s["attention_mask"][:,-64:] for s in prose_sample_toks]
            prose_task_vector = torch.zeros(768).to(device)
            for batch_ids, batch_mask in zip(torch.split(torch.cat(sample_ids), 4),
                                             torch.split(torch.cat(sample_mask), 4)):
                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                batch_mean = vae_model.encode(batch_ids, batch_mask)
                batch_z = vae_model.vae.sample(batch_mean)
                for z in batch_z:
                    prose_task_vector += z
            if context:
                context_toks = tokenizer(context, return_tensors="pt")
                context_ids = context_toks["input_ids"].to(device)
                context_mask = context_toks["attention_mask"].to(device)
            else:
                context_ids = torch.empty([1,0]).long().to(device)
                context_mask = torch.empty([1,0]).long().to(device)
            topic_toks = tokenizer(topic, return_tensors="pt")
            embed_ids = topic_toks["input_ids"].to(device)
            embed_mask = topic_toks["attention_mask"].to(device)
            mean = vae_model.encode(embed_ids, embed_mask)
            topic_embed = vae_model.vae.sample(mean)
            paragraph_sentences = []
            for i in range(5):
                mean = vae_model.encode(embed_ids, embed_mask)
                z = vae_model.vae.sample(mean)
                embeds = []
                for i in range(n_avg):
                    task_z = (z * 0.7 + topic_embed * 0.2 + prose_task_vector * 0.1)
                    output_ids = router.generate(task_z,
                                                 context_ids,
                                                 context_mask,
                                                 128,
                                                 tau=0.9)
                    print(tokenizer.decode(output_ids[0][-130:], skip_special_tokens=True))
                    intermediate_embed_ids = output_ids[:,-64:]
                    intermediate_embed_mask = context_mask.new_ones(
                        [1, intermediate_embed_ids.shape[1]]
                    )
                    mean = vae_model.encode(intermediate_embed_ids, intermediate_embed_mask)
                    embeds.append(vae_model.vae.sample(mean))
                z = ((sum(embeds) / n_avg * 0.8) + prose_task_vector * 0.2)
                output_ids = router.generate(z,
                                             context_ids,
                                             context_mask,
                                             128,
                                             tau=0.9)
                context_ids = torch.cat([context_ids, embed_ids], dim=1)
                context_mask = torch.cat([context_mask, embed_mask], dim=1)
                paragraph_sentences.append(z)
                embed_ids = output_ids[:,-128:-64]
                embed_mask = context_mask.new_ones([1, embed_ids.shape[1]])
            next_topic_z = (paragraph_sentences[0] * 0.5
                            + paragraph_sentences[-1] * 0.3
                            + (sum(paragraph_sentences[1:-1]) * 0.2 / len(paragraph_sentences[1:-1])))
            break_toks = tokenizer("\n\n", return_tensors="pt")
            output_ids = router.generate(next_topic_z,
                                         torch.cat([context_ids,
                                                    break_toks["input_ids"].to(device)], dim=1),
                                         torch.cat([context_mask,
                                                    break_toks["attention_mask"].to(device)], dim=1),
                                         128,
                                         tau=1)
            out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in context_ids]
            return out_texts, tokenizer.decode(output_ids[0][-130:], skip_special_tokens=True)
        
    def bigvae_generate_plan(vae_model, router, terminal=None, n_steps=5, n_avg=4, start=None, context=None):
        ae_scale = 1.28125
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            prose_sample_toks = [tokenizer(sample,
                                           return_tensors="pt",
                                           add_special_tokens=False)
                                 for sample in prose_samples]
            sample_ids = [s["input_ids"][:,:-64] for s in prose_sample_toks]
            sample_mask = [s["attention_mask"][:,:-64] for s in prose_sample_toks]
            prose_task_vector = torch.zeros(768).to(device)
            for batch_ids, batch_mask in zip(torch.split(torch.cat(sample_ids), 4),
                                             torch.split(torch.cat(sample_mask), 4)):
                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                batch_mean = vae_model.encode(batch_ids, batch_mask)
                batch_z = vae_model.vae.sample(batch_mean)
                for z in batch_z:
                    prose_task_vector += z
            prose_task_vector /= torch.cat(sample_ids).shape[0]
            prose_task_vector *= 2.5
            if context:
                context_toks = tokenizer(context, return_tensors="pt")
                context_ids = context_toks["input_ids"].to(device)
                context_mask = context_toks["attention_mask"].to(device)
            else:
                context_ids = torch.empty([1,0]).long().to(device)
                context_mask = torch.empty([1,0]).long().to(device) 
            if start:
                start_toks = tokenizer(start, return_tensors="pt")
                start_ids = start_toks["input_ids"].to(device)
                start_mask = start_toks["attention_mask"].to(device)
                mean = vae_model.encode(start_ids, start_mask)
                z = vae_model.vae.sample(mean) * 0.8 + prose_task_vector * 0.2
            else:
                embed = (torch.randn([1, args.z_dim]) * ae_scale).to(device)
                z = (embed * 0.7 + prose_task_vector * 0.3)
                z *= (50 / z.norm().item())
            if terminal:
                terminal_toks = tokenizer(terminal,
                                          return_tensors="pt",
                                          add_special_tokens=False)
                terminal_ids = terminal_toks["input_ids"].to(device)
                terminal_mask = terminal_toks["attention_mask"].to(device)
                mean = vae_model.encode(terminal_ids, terminal_mask)
                terminal_embed = vae_model.vae.sample(mean)
            else:
                embeds = []
                for i in range(n_avg):
                    embed_ids, embed_mask = bigvae_generate_task(vae_model, router, start)
                    mean = vae_model.encode(embed_ids[-64:], embed_mask[-64:])
                    embeds.append(vae_model.vae.sample(mean))
                terminal_embed = torch.mean(torch.cat(embeds, dim=0), dim=0).unsqueeze(0)
                if terminal_embed.norm().item() < 50:
                    terminal_embed *= (50 / terminal_embed.norm().item())
            for step in torch.tensor([i for i in range(1, n_steps+1)]) * 0.05:
                embeds = []
                for i in range(n_avg):
                    output_ids = router.generate(z,
                                                 context_ids,
                                                 context_mask,
                                                 128,
                                                 tau=0.9)
                    print(tokenizer.decode(output_ids[0][-128:]))
                    intermediate_embed_ids = output_ids[:,-64:]
                    intermediate_embed_mask = context_mask.new_ones(
                        [1, intermediate_embed_ids.shape[1]]
                    )
                    mean = vae_model.encode(intermediate_embed_ids, intermediate_embed_mask)
                    embeds.append(vae_model.vae.sample(mean))
                avg_z = (sum(embeds) / n_avg * (0.95-step)) + terminal_embed * (0.05+step)
                # avg_z = (sum(embeds) / n_avg * 0.9) + terminal_embed * 0.1
                avg_z *= (50 / avg_z.norm().item())
                output_ids = router.generate(avg_z,
                                             context_ids,
                                             context_mask,
                                             64,
                                             tau=0.9)
                print(tokenizer.decode(output_ids[0][-128:]))
                if start:
                    context_ids = torch.cat([context_ids, embed_ids], dim=1)
                    context_mask = torch.cat([context_mask, embed_mask], dim=1)
                embed_ids = output_ids[:,-64:]
                embed_mask = context_mask.new_ones([1, embed_ids.shape[1]])
                if not start:
                    context_ids = torch.cat([context_ids, embed_ids], dim=1)
                    context_mask = torch.cat([context_mask, embed_mask], dim=1)
                mean = vae_model.encode(embed_ids, embed_mask)
                z = vae_model.vae.sample(mean)
            out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in context_ids]
            return out_texts  

        
    def bigvae_generate_plan_2(vae_model, router, terminal=None, n_steps=5, n_avg=4, start=None, context=None):
        ae_scale = 1.28125
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            prose_sample_toks = [tokenizer(sample,
                                           return_tensors="pt",
                                           add_special_tokens=False)
                                 for sample in prose_samples]
            sample_ids = [s["input_ids"][:,:64] for s in prose_sample_toks]
            sample_mask = [s["attention_mask"][:,:64] for s in prose_sample_toks]
            prose_task_vector = torch.zeros(768).to(device)
            total_norm = 0
            for batch_ids, batch_mask in zip(torch.split(torch.cat(sample_ids), 4),
                                             torch.split(torch.cat(sample_mask), 4)):
                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                batch_mean = vae_model.encode(batch_ids, batch_mask)
                batch_z = vae_model.vae.sample(batch_mean)
                for z in batch_z:
                    prose_task_vector += z
                    total_norm += z.norm().item()
            prose_task_vector /= torch.cat(sample_ids).shape[0]
            avg_norm = total_norm / torch.cat(sample_ids).shape[0]
            prose_task_vector *= (avg_norm / prose_task_vector.norm().item())
            if context:
                context_toks = tokenizer(context, return_tensors="pt")
                context_ids = context_toks["input_ids"].to(device)
                context_mask = context_toks["attention_mask"].to(device)
            else:
                context_ids = torch.empty([1,0]).long().to(device)
                context_mask = torch.empty([1,0]).long().to(device) 
            if start:
                start_toks = tokenizer(start, return_tensors="pt")
                start_ids = start_toks["input_ids"].to(device)
                start_mask = start_toks["attention_mask"].to(device)
                mean = vae_model.encode(start_ids, start_mask)
                z = vae_model.vae.sample(mean) * 0.8 + prose_task_vector * 0.2
            else:
                embed = (torch.randn([1, args.z_dim]) * ae_scale).to(device)
                z = (embed * 0.7 + prose_task_vector * 0.3)
                z *= (33 / z.norm().item())
            if terminal:
                terminal_toks = tokenizer(terminal,
                                          return_tensors="pt",
                                          add_special_tokens=False)
                terminal_ids = terminal_toks["input_ids"].to(device)
                terminal_mask = terminal_toks["attention_mask"].to(device)
                mean = vae_model.encode(terminal_ids, terminal_mask)
                terminal_embed = vae_model.vae.sample(mean)
            else:
                embeds = []
                for i in range(n_avg):
                    embed_ids, embed_mask = bigvae_generate_task(vae_model, router, start)
                    mean = vae_model.encode(embed_ids[-64:], embed_mask[-64:])
                    embeds.append(vae_model.vae.sample(mean))
                terminal_embed = torch.mean(torch.cat(embeds, dim=0), dim=0).unsqueeze(0)
                if terminal_embed.norm().item() < 33:
                    avg_norm = sum([e.norm() for e in embeds]) / len(embeds)
                    terminal_embed *= (avg_norm / terminal_embed.norm().item())
            for step in torch.tensor([i for i in range(1, n_steps+1)]) * 0.05:
                z = z * (0.95-step) + terminal_embed * (0.05+step)
                # avg_z = (sum(embeds) / n_avg * 0.9) + terminal_embed * 0.1
                if z.norm().item() < 33:
                    z *= (33 / z.norm().item())
                output_ids = router.generate(z,
                                             context_ids,
                                             context_mask,
                                             64,
                                             tau=0.9)
                print(tokenizer.decode(output_ids[0][-128:]))
                if start:
                    context_ids = torch.cat([context_ids, embed_ids], dim=1)
                    context_mask = torch.cat([context_mask, embed_mask], dim=1)
                embed_ids = output_ids[:,-64:]
                embed_mask = context_mask.new_ones([1, embed_ids.shape[1]])
                if not start:
                    context_ids = torch.cat([context_ids, embed_ids], dim=1)
                    context_mask = torch.cat([context_mask, embed_mask], dim=1)
                mean = vae_model.encode(embed_ids, embed_mask)
                z = vae_model.vae.sample(mean)
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
        
    for i in range(3):
        context, prompt = bigvae_generate_paragraph_topic(vae_model, router, prompt, context=context)
    print(context)

if __name__ == "__main__":
    main()
