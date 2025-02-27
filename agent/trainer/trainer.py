#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import peft
import torch
from torch import distributed as dist, optim
from torch.nn import functional as F
from torch.utils import data
import torch_dist_utils as du
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from linear_4bit_sharded import quantize_and_shard
from patch_model import patch_model

print = tqdm.external_write_mode()(print)
print0 = tqdm.external_write_mode()(du.print0)


class Dataset(data.Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = Path(path).read_text().splitlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.tokenizer(json.loads(self.dataset[idx])["text"]).input_ids


class CollateFn:
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, batch):
        input_ids = torch.full((len(batch), self.seq_len), 0, dtype=torch.long)
        target_ids = torch.full((len(batch), self.seq_len), -100, dtype=torch.long)
        for i, x in enumerate(batch):
            ids = torch.tensor(x, dtype=torch.long)
            max_len = min(len(x) - 1, self.seq_len)
            input_ids[i, :max_len] = ids[:max_len]
            target_ids[i, :max_len] = ids[1 : max_len + 1]
        return input_ids, target_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per group")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length")
    args = parser.parse_args()

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_group = du.get_local_group()
    local_rank = dist.get_rank(local_group)
    local_world_size = dist.get_world_size(local_group)
    group_rank = int(os.environ["GROUP_RANK"])
    group_world_size = world_size // local_world_size
    seq_len_device = args.seq_len // local_world_size

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = Dataset(args.dataset, tokenizer)
    sampler = data.DistributedSampler(
        dataset, group_world_size, group_rank, shuffle=True, seed=1234, drop_last=True
    )
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=CollateFn(args.seq_len),
    )

    patch_model(local_group)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = quantize_and_shard(model, device, local_group)
    torch.cuda.empty_cache()
    peft_config = peft.LoraConfig(
        peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    model = peft.get_peft_model(model, peft_config)
    du.broadcast_tensors(p for p in model.parameters() if p.requires_grad)

    if rank == 0:
        model.print_trainable_parameters()

    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    opt = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

    for i, (input_ids, target_ids) in enumerate(tqdm(dataloader, disable=rank != 0)):
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        seq_start = local_rank * seq_len_device
        seq_end = (local_rank + 1) * seq_len_device
        input_ids_local = input_ids[:, seq_start:seq_end]
        target_ids_local = target_ids[:, seq_start:seq_end]
        position_ids_local = torch.arange(seq_start, seq_end, device=device)
        position_ids_local = position_ids_local.expand_as(input_ids_local)
        n_targets = torch.sum(target_ids_local != -100)
        dist.all_reduce(n_targets)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids_local, position_ids=position_ids_local, use_cache=False).logits
            loss = F.cross_entropy(logits.mT, target_ids_local, reduction="sum") / n_targets
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        handles = [dist.all_reduce(g, async_op=True) for g in grads]
        for handle in handles:
            handle.wait()
        opt.step()
        opt.zero_grad()
        dist.all_reduce(loss)
        print0(f"step: {i}, loss: {loss:g}")

    if rank == 0:
        model.save_pretrained(args.output, safe_serialization=True)
    dist.barrier()


if __name__ == "__main__":
    main()
