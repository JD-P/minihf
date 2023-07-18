#!/usr/bin/env python3

"""Train a MiniHF evaluator model (instruction tuned LoRA)."""

import argparse
from functools import partial
import os
from pathlib import Path
import sys

os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import accelerate
import datasets
import datasets.distributed
import peft
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

print = tqdm.external_write_mode()(print)


def batch_to_tensors(batch, device="cpu"):
    batch = [item["input_ids"] for item in batch]
    seq_len = max(len(x) for x in batch)
    input_ids = torch.zeros(len(batch), seq_len, dtype=torch.long, device=device)
    attention_mask = torch.zeros(len(batch), seq_len, dtype=torch.long, device=device)
    for i, x in enumerate(batch):
        input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long, device=device)
        attention_mask[i, : len(x)] = 1
    return input_ids, attention_mask


def weighted_mean(x, w=None, dim=None, keepdim=False, dtype=None):
    w = x.new_tensor(1.0) if w is None else w
    w = w.expand_as(x)
    dim = tuple(range(x.ndim)) if dim is None else dim
    num = torch.sum(x * w, dim=dim, keepdim=keepdim, dtype=dtype)
    denom = torch.sum(w, dim=dim, keepdim=keepdim, dtype=dtype)
    return num / denom


class EndlessHFDataset(data.IterableDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        while True:
            yield from self.dataset
            self.dataset.set_epoch(self.dataset._epoch + 1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch-size", type=int, default=4, help="batch size per process")
    parser.add_argument("--examples", type=int, default=100000, help="train for n examples")
    parser.add_argument("--output-dir", type=Path, default="evaluator", help="output directory")
    parser.add_argument("--save-every", type=int, default=10000, help="save every n examples")
    args = parser.parse_args()

    dataset_seed = 100
    lora_rank = 32
    lr = 1e-4
    max_len = 2048
    model_name = "openlm-research/open_llama_7b"

    # Initialize Accelerate
    accelerator = accelerate.Accelerator(mixed_precision="bf16", dispatch_batches=False)
    device = accelerator.device
    print0 = accelerator.on_local_main_process(print)

    # Load tokenizer
    print0(f"### Loading tokenizer: {model_name}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print0(f"### Loading model: {model_name}", file=sys.stderr)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    with accelerator.main_process_first():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if accelerator.num_processes == 1 else {"": device},
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    accelerator.wait_for_everyone()

    # Set up the LoRA
    print0("### Setting up the LoRA", file=sys.stderr)
    peft_config = peft.LoraConfig(
        peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "lm_head",
        ],
    )
    model = peft.get_peft_model(model, peft_config)
    accelerator.wait_for_everyone()

    # Set up the model
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if accelerator.is_local_main_process:
        model.print_trainable_parameters()

    # Dataset helper functions
    def combine_flan(row):
        return row["inputs"] + "<|end|>" + row["targets"] + tokenizer.eos_token

    def combine_dolly(row):
        return (
            row["context"]
            + "\n\n"
            + row["instruction"]
            + "<|end|>"
            + row["response"]
            + tokenizer.eos_token
        )

    def to_tokens(combine_fn, row):
        return tokenizer(combine_fn(row))

    def exclude_too_long(row):
        return len(row["input_ids"]) <= max_len

    # Load dataset
    print0("### Loading datasets", file=sys.stderr)
    with accelerator.main_process_first():
        dataset_1 = datasets.load_dataset("Muennighoff/flan", streaming=True)
        dataset_2 = datasets.load_dataset("databricks/databricks-dolly-15k", streaming=True)
    accelerator.wait_for_everyone()
    dataset_1 = dataset_1["train"].map(partial(to_tokens, combine_flan))
    dataset_2 = dataset_2["train"].map(partial(to_tokens, combine_dolly))
    dataset = (
        datasets.interleave_datasets([dataset_1, dataset_2], probabilities=[0.9, 0.1])
        .filter(exclude_too_long)
        .shuffle(seed=dataset_seed)
        .select_columns(["input_ids"])
    )
    dataset = datasets.distributed.split_dataset_by_node(
        dataset, accelerator.process_index, accelerator.num_processes
    )
    dataloader = data.DataLoader(
        EndlessHFDataset(dataset),
        batch_size=args.batch_size,
        collate_fn=batch_to_tensors,
        drop_last=True,
    )

    # Set up optimizer
    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    # Wrap objects
    model, opt, dataloader = accelerator.prepare(model, opt, dataloader)

    # Test max sequence length
    print0("### Testing max sequence length", file=sys.stderr)
    input_ids = torch.zeros([args.batch_size, max_len], dtype=torch.long, device=device)
    attention_mask = torch.ones([args.batch_size, max_len], dtype=torch.long, device=device)
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
    accelerator.backward(outputs.logits.sum() * 0)
    opt.zero_grad()
    torch.cuda.empty_cache()

    def save_model():
        print0("### Saving model", file=sys.stderr)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)

    # Train
    print0("### Training", file=sys.stderr)
    examples = 0
    last_save = 0
    pbar = tqdm(
        disable=not accelerator.is_local_main_process,
        total=args.examples,
        unit="ex",
        smoothing=0.01,
    )

    try:
        for batch in dataloader:
            input_ids, attention_mask = batch
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(
                    input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    use_cache=False,
                )
                losses = F.cross_entropy(
                    outputs.logits.transpose(-1, -2),
                    input_ids[:, 1:],
                    reduction="none",
                )
                mask = attention_mask[:, :-1] * attention_mask[:, 1:]
                loss = weighted_mean(losses, mask, dtype=torch.float32)

                # Backward pass and optimizer step
                accelerator.backward(loss)
                opt.step()
                opt.zero_grad()

            global_batch_size = args.batch_size * accelerator.num_processes
            examples += global_batch_size
            pbar.update(global_batch_size)

            global_loss = accelerator.reduce(loss, "mean")
            print0(f"examples: {examples}, loss: {global_loss.item():g}")

            if examples >= args.examples:
                save_model()
                break

            if examples - last_save >= args.save_every:
                save_model()
                last_save += args.save_every

    except KeyboardInterrupt:
        pass

    finally:
        pbar.close()


if __name__ == "__main__":
    main()
