#!/usr/bin/env python3

import torch
from torch import distributed as dist
from torch.distributed import nn as dnn
import torch_dist_utils as du
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mixtral_ring_attn import patch_model


def kl_divergence(logits_p, logits_q):
    logp = torch.nn.functional.log_softmax(logits_p, dim=-1)
    logq = torch.nn.functional.log_softmax(logits_q, dim=-1)
    return torch.sum(torch.exp(logp) * (logp - logq), dim=-1)


def main():
    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model_name = "mistralai/Mixtral-8x7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "The quick brown fox jumps over the lazy dog, " * 8
    tokens = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"][:, :64]
    n_tokens = tokens.shape[1]
    assert n_tokens % world_size == 0
    n_tokens_device = n_tokens // world_size
    du.print0("Number of tokens:", n_tokens)
    du.print0("Number of tokens per device:", n_tokens_device)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2",
        )
        .eval()
        .requires_grad_(False)
    )
    logits_ref = model(tokens, use_cache=False).logits

    del model
    torch.cuda.empty_cache()
    patch_model()

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2",
        )
        .eval()
        .requires_grad_(False)
    )
    tokens_device = tokens[:, rank * n_tokens_device : (rank + 1) * n_tokens_device]
    logits = model(tokens_device, use_cache=False).logits
    logits_all = torch.cat(dnn.all_gather(logits), dim=1)

    # error = torch.sqrt(torch.mean((logits_all - logits_ref) ** 2))
    error = kl_divergence(logits_ref, logits_all).mean()
    with du.do_in_order():
        print(f"Rank {rank}: error = {error}")


if __name__ == "__main__":
    main()
