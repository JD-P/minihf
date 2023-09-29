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
        with set_adapter(self.model, "encoder"), disable_causal_mask():
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
        safetorch.save_file(self.vae.vae.state_dict(), path / "vae.safetensors")

    def load_pretrained(self, path, is_trainable=False):
        path = Path(path)
        self.model.delete_adapter("router")
        try:
            self.model.load_adapter(path / "router", "router", is_trainable=is_trainable)
        except:
            self.model.load_adapter(path / "decoder", "router", is_trainable=is_trainable)
        
    def encode(self, input_ids, attention_mask):
        with set_adapter(self.vae.model, "encoder"), disable_causal_mask():
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
    parser.add_argument("--preprocessed", type=str, help="preprocessed dataset dir")
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
        default="openlm-research/open_llama_3b_v2",
        help="model name",
    )
    parser.add_argument("--context", type=int, default=2048, help="context window length")
    parser.add_argument("--vae-context", type=int, default=128, help="vae embed context")
    parser.add_argument("--output", type=Path, required=True, help="path to save adapter")
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
        # router.load_pretrained(args.start_from, is_trainable=True)
    accelerator.wait_for_everyone()

    router.train()
    if args.gradient_checkpointing:
        router.model.gradient_checkpointing_enable()
        router.model.enable_input_require_grads()

    if is_main:
        router.model.print_trainable_parameters()

    router.model.set_adapter("router")
    opt = optim.Adam(router.model.parameters(),
                     lr=args.lr,
                     betas=(0.9, 0.99))

    input_ids_all, attention_mask_all = [], []
    for shard_name in os.listdir(args.preprocessed):
        data_path = os.path.join(args.preprocessed, shard_name)
        data_file = safetorch.load_file(data_path)
        input_ids = torch.split(data_file["input_ids"], args.context, dim=1)
        attention_mask = torch.split(data_file["attention_mask"], args.context, dim=1)
        if input_ids[-1].shape[1] != args.context:
            input_ids = input_ids[:-1]
            attention_mask = attention_mask[:-1]
        input_ids_all.extend(input_ids)
        attention_mask_all.extend(attention_mask)
    del data_file, input_ids, attention_mask
    input_ids_all = torch.cat(input_ids_all)
    attention_mask_all = torch.cat(attention_mask_all)
    valid_indices = attention_mask_all.sum(dim=1) == args.context
    input_ids_all = input_ids_all[valid_indices]
    attention_mask_all = attention_mask_all[valid_indices]
    del valid_indices

    preprocessed = data.TensorDataset(input_ids_all, attention_mask_all)

    dataloader = data.DataLoader(
        preprocessed,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    router, opt, dataloader = accelerator.prepare(router, opt, dataloader)

    i = 0
    kl_sched = cosine_warmup(5000, 0.01)

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def demo(model, input_ids, attention_mask, n_tokens):
        bs = min(input_ids.shape[0], 2)
        n_outputs = 2
        tau = 0.8

        index = random.randrange(args.context - (args.vae_context * 2))
        context_ids = input_ids[:,:index]
        context_mask = attention_mask[:,:index]
        embed_ids = input_ids[:,index:index + args.vae_context]
        embed_mask = attention_mask[:,index:index + args.vae_context]
        target_ids = input_ids[:,index:index + args.vae_context * 2]
        target_mask = input_ids[:,index:index + args.vae_context * 2]

        in_texts = [tokenizer.decode(toks, skip_special_tokens=True)
                    for toks in torch.cat([context_ids, embed_ids], dim=1)]
        mean = model.encode(embed_ids[:bs], embed_mask[:bs])
        z = model.vae.vae.sample(mean.repeat_interleave(n_outputs, 0), tau=tau)
        context_ids = context_ids[:bs].repeat_interleave(n_outputs, 0)
        context_mask = context_mask[:bs].repeat_interleave(n_outputs, 0)
        # empty = z.new_zeros([z.shape[0], 0], dtype=torch.long)
        output_ids = model.generate(z, context_ids, context_mask, n_tokens, tau=tau)
        out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in output_ids]
        out_texts = list(batched(out_texts, n_outputs))
        print("======")
        for in_text, out_batch in zip(in_texts, out_texts):
            print("=== Input ===")
            print(in_text)
            print("=== Outputs ===")
            for i, out_text in enumerate(out_batch):
                print(out_text)
                if i < len(out_batch) - 1:
                    print("===")
        print("======")

    def save():
        print0(f"### Saving model to {args.output}", file=sys.stderr)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(router)
            unwrapped_model.save_pretrained(args.output)
            state_obj = {"step": i, "last_kl_weight": kl_sched(i)}
            with open(args.output / "state.json", "w") as f:
                json.dump(state_obj, f)

    accelerator.wait_for_everyone()
    for epoch in trange(args.epochs, disable=not is_main):
        for input_ids, attention_mask in tqdm(dataloader, disable=not is_main):
            input_ids = input_ids.long()
            if is_main and i % 100 == 0:
                demo(accelerator.unwrap_model(router), input_ids, attention_mask, args.vae_context)
                pass
            with accelerator.accumulate(router):
                index = random.randrange(args.context - (args.vae_context * 2))
                context_ids = input_ids[:,:index]
                context_mask = attention_mask[:,:index]
                embed_ids = input_ids[:,index:index + args.vae_context]
                embed_mask = attention_mask[:,index:index + args.vae_context]
                target_ids = input_ids[:,index:index + args.vae_context * 2]
                target_mask = attention_mask[:,index:index + args.vae_context * 2]

                drop_mask = torch.rand([context_ids.shape[0], 1], device=device) < 0.5
                context_ids = torch.where(drop_mask, torch.zeros_like(context_ids), context_ids)
                context_mask = torch.where(drop_mask, torch.zeros_like(context_mask), context_mask)
                outputs = router(embed_ids, embed_mask,
                                 target_ids[:,:-1], target_mask[:,:-1],
                                 context_ids, context_mask)
                rec_losses = F.cross_entropy(
                    outputs.logits[:, -args.vae_context * 2:].transpose(-1, -2),
                    target_ids,
                    reduction="none",
                )
                n_toks = target_mask.sum()
                rec_loss = torch.sum(rec_losses * target_mask, dtype=torch.float32) / n_toks
                # kl_loss = torch.sum(mean**2 / 2, dtype=torch.float32) * kl_sched(i) / n_toks
                loss = rec_loss # + kl_loss

                # accelerator.backward(loss, inputs=list(p for p in accelerator.unwrap_model(router).model.parameters() if p.requires_grad))
                accelerator.backward(loss)
                # for n, p in router.named_parameters():
                #     if p.grad is not None:
                #         grad_norm = torch.norm(p.grad, dtype=torch.float32)
                #         if grad_norm != 0:
                #             print(f"{n}: {grad_norm:g}", file=sys.stderr)
                opt.step()
                opt.zero_grad()

                loss_global, rec_global = accelerator.reduce(
                    (loss, rec_loss), "mean"
                )
                print0(
                    f"epoch: {epoch}, step: {i}, loss: {loss_global.item():g}, rec: {rec_global.item():g}",
                    file=sys.stderr,
                )
                i += 1

                if i % args.save_every == 0:
                    save()

        save()


if __name__ == "__main__":
    main()
