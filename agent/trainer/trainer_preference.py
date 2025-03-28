#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import peft
import torch
from torch import distributed as dist, optim
from torch.nn import functional as F
from torch.distributed import nn as dnn
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
        item = json.loads(self.dataset[idx])
        prompt = self.tokenizer(item["prompt"]).input_ids
        completions = [
            self.tokenizer(cmpl["completion"], add_special_tokens=False).input_ids
            for cmpl in item["completions"]
        ]
        rewards = [cmpl["reward"] for cmpl in item["completions"]]
        return prompt, completions, rewards


class CollateFn:
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, batch):
        prompt, completions, rewards = zip(*batch)
        n = sum(len(cmpl) for cmpl in completions)
        input_ids = torch.full((n, self.seq_len), 0, dtype=torch.long)
        logp_mask = torch.full((n, self.seq_len), False, dtype=torch.bool)
        reward = torch.full((n,), 0, dtype=torch.float32)
        groups = torch.full((n,), 0, dtype=torch.long)
        i = 0
        for group, (pr, cmpl, r) in enumerate(zip(prompt, completions, rewards)):
            ids_all = [torch.tensor(pr + c, dtype=torch.long) for c in cmpl]
            for ids, r_i in zip(ids_all, r):
                max_len = min(len(ids), self.seq_len)
                input_ids[i, :max_len] = ids[:max_len]
                logp_mask[i, len(pr) : max_len] = True
                groups[i] = group
                reward[i] = r_i
                i += 1
        return input_ids, logp_mask, reward, groups


def logp_completion_distributed(logits, tokens, mask, group=None):
    logits = F.log_softmax(logits, dim=-1)
    logp_tokens = logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    logp = torch.sum(logp_tokens * mask, dim=-1)
    return dnn.all_reduce(logp, group=group)


def spo_loss(logp, logp_ref, reward, groups, beta):
    """Compute the Scalar Preference Optimization loss.

    The SPO loss takes as input groups of log probabilities of completions given the same prompt
    for each completion in a group, under the model and a reference model, and scalar rewards for
    each completion. It regresses the difference between the model's implied rewards for the
    completions toward the difference between their actual rewards, scaled by the inverse of the
    KL penalty coefficient.

    Args:
        logp: Log probabilities of the completions given their prompts under the
            model. Should be differentiable w.r.t. the model parameters. Shape: (N).
        logp_ref: Log probabilities of the completions given their prompts under the
            reference model. Shape: (N).
        reward: Rewards for the completions. Shape: (N).
        groups: Group indices for the completions. Shape: (N). A tensor of integers where each
            integer indicates the group to which the corresponding completion belongs. The groups
            should be numbered from 0 to the number of groups minus 1, and each group should have
            at least two members.
        beta: The KL penalty coefficient.

    Returns:
        The Scalar Preference Optimization loss (sum reduction).
    """
    n = groups.bincount()[groups]
    size = groups.amax() + 1
    ir = logp - logp_ref
    mean_ir = ir.new_zeros(size).index_add_(0, groups, ir)[groups] / n
    mean_reward = reward.new_zeros(size).index_add_(0, groups, reward)[groups] / n
    scale = 0.5 * beta * n / (n - 1)
    parts = scale * ((ir - mean_ir) - (reward - mean_reward) / beta) ** 2
    return torch.sum(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--reference", type=str, required=True, help="SFT model name or path")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per group")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="KL penalty weight")
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
    model = peft.PeftModel.from_pretrained(model, args.reference, adapter_name="reference")
    model.load_adapter(args.reference, adapter_name="default", is_trainable=True)

    if rank == 0:
        model.print_trainable_parameters()

    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    opt = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

    for i, (input_ids, logp_mask, reward, groups) in enumerate(tqdm(dataloader, disable=rank != 0)):
        input_ids = input_ids.to(device)
        logp_mask = logp_mask.to(device)
        reward = reward.to(device)
        groups = groups.to(device)

        seq_start = local_rank * seq_len_device
        seq_end = (local_rank + 1) * seq_len_device
        input_ids_local = input_ids[:, seq_start:seq_end]
        target_ids_local = input_ids[:, seq_start + 1 : seq_end + 1]  # SHIFTED LEFT
        logp_mask_local = logp_mask[:, seq_start + 1 : seq_end + 1]  # SHIFTED LEFT
        position_ids_local = torch.arange(seq_start, seq_end, device=device)
        position_ids_local = position_ids_local.expand_as(input_ids_local)
        loss_constant = args.batch_size * world_size  # there is probably a better way to do this

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.set_adapter("reference")
            with torch.no_grad():
                logits_ref = model(
                    input_ids_local, position_ids=position_ids_local, use_cache=False
                ).logits
                logp_ref = logp_completion_distributed(
                    logits_ref, target_ids_local, logp_mask_local, local_group
                )
            model.set_adapter("default")
            logits = model(input_ids_local, position_ids=position_ids_local, use_cache=False).logits
            logp = logp_completion_distributed(
                logits, target_ids_local, logp_mask_local, local_group
            )
            loss = spo_loss(logp, logp_ref, reward, groups, args.kl_weight) / loss_constant

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
        model.save_pretrained(args.output, safe_serialization=True, selected_adapters=["default"])
    dist.barrier()


if __name__ == "__main__":
    main()
