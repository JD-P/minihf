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
    def __init__(self, model_name, device, z_dim=768, lora_rank=32, dropout=0.0):
        super().__init__()
        self.device = device
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
        peft_config = peft.LoraConfig(
            peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=8,
            lora_dropout=dropout,
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
        self.model = peft.get_peft_model(model, peft_config, "encoder")
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
    parser.add_argument("--context", type=int, default=64, help="context window length")
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
        model = DecoderOnlyTransformerVAE(
            model_name, device, z_dim=args.z_dim, lora_rank=args.rank, dropout=args.dropout
        )
    accelerator.wait_for_everyone()

    model.train()
    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()
        model.model.enable_input_require_grads()

    if is_main:
        model.model.print_trainable_parameters()

    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    input_ids_all, attention_mask_all = [], []
    for shard_name in os.listdir(args.preprocessed):
        data_path = os.path.join(args.preprocessed, shard_name)
        data_file = safetorch.load_file(data_path)
        input_ids = torch.split(data_file["input_ids"], args.context * 2, dim=1)
        attention_mask = torch.split(data_file["attention_mask"], args.context * 2, dim=1)
        if input_ids[-1].shape[1] != args.context * 2:
            input_ids = input_ids[:-1]
            attention_mask = attention_mask[:-1]
        input_ids_all.extend(input_ids)
        attention_mask_all.extend(attention_mask)
    del data_file, input_ids, attention_mask
    input_ids_all = torch.cat(input_ids_all)
    attention_mask_all = torch.cat(attention_mask_all)
    valid_indices = attention_mask_all.sum(dim=1) > args.context
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

    model, opt, dataloader = accelerator.prepare(model, opt, dataloader)

    i = 0
    kl_sched = cosine_warmup(5000, 0.01)

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def demo(model, input_ids, attention_mask, n_tokens):
        bs = min(input_ids.shape[0], 4)
        n_outputs = 4
        tau = 0.1

        input_ids_1, input_ids_2 = input_ids.chunk(2, dim=1)
        attn_mask_1, attn_mask_2 = attention_mask.chunk(2, dim=1)

        in_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in input_ids]
        mean = model.encode(input_ids_2[:bs], attn_mask_2[:bs])
        z = model.vae.sample(mean.repeat_interleave(n_outputs, 0), tau=tau)
        input_ids_1 = input_ids_1[:bs].repeat_interleave(n_outputs, 0)
        attn_mask_1 = attn_mask_1[:bs].repeat_interleave(n_outputs, 0)
        # empty = z.new_zeros([z.shape[0], 0], dtype=torch.long)
        output_ids = model.generate(z, input_ids_1, attn_mask_1, n_tokens, tau=tau)
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
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output)
            state_obj = {"step": i, "last_kl_weight": kl_sched(i)}
            with open(args.output / "state.json", "w") as f:
                json.dump(state_obj, f)

    accelerator.wait_for_everyone()
    for epoch in trange(args.epochs, disable=not is_main):
        for input_ids, attention_mask in tqdm(dataloader, disable=not is_main):
            input_ids = input_ids.long()
            if is_main and i % 100 == 0:
                demo(accelerator.unwrap_model(model), input_ids, attention_mask, args.context)

            with accelerator.accumulate(model):
                input_ids_1, input_ids_2 = input_ids.chunk(2, dim=1)
                attn_mask_1, attn_mask_2 = attention_mask.chunk(2, dim=1)

                drop_mask = torch.rand([input_ids_1.shape[0], 1], device=device) < 0.5
                input_ids_1 = torch.where(drop_mask, torch.zeros_like(input_ids_1), input_ids_1)
                attn_mask_1 = torch.where(drop_mask, torch.zeros_like(attn_mask_1), attn_mask_1)

                outputs, mean = model(
                    input_ids_2[:, :-1], attn_mask_2[:, :-1], input_ids_1, attn_mask_1
                )
                rec_losses = F.cross_entropy(
                    outputs.logits[:, args.context :].transpose(-1, -2),
                    input_ids_2,
                    reduction="none",
                )
                assert (args.context % 2) == 0
                tail_len = int(args.context / 4 / 2)
                attn_mask_2 = torch.cat([attn_mask_2[:,:tail_len] * 0.05,
                                         attn_mask_2[:,tail_len:len(attn_mask_2[0])-tail_len],
                                         attn_mask_2[:,-tail_len:] * 0.15], dim=1)
                n_toks = attn_mask_2.sum()
                rec_loss = torch.sum(rec_losses * attn_mask_2, dtype=torch.float32) / n_toks
                kl_loss = torch.sum(mean**2 / 2, dtype=torch.float32) * kl_sched(i) / n_toks
                loss = rec_loss + kl_loss

                accelerator.backward(loss)
                opt.step()
                opt.zero_grad()

                loss_global, rec_global, kl_global = accelerator.reduce(
                    (loss, rec_loss, kl_loss), "mean"
                )
                print0(
                    f"epoch: {epoch}, step: {i}, loss: {loss_global.item():g}, rec: {rec_global.item():g}, kl: {kl_global.item():g}",
                    file=sys.stderr,
                )
                i += 1

                if i % args.save_every == 0:
                    save()

        save()


if __name__ == "__main__":
    main()
