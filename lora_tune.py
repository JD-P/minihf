#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import os
import sys

import peft
import torch
from torch import nn, optim
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import trange, tqdm
from dataset import ZippedConversationsDataset

print = tqdm.external_write_mode()(print)


def lora_tune_evaluator(data, continue_from=None):
    if continue_from:
        peft_config = peft.PeftConfig.from_pretrained(os.path.join("reward_models/", continue_from))
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model_base = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map="sequential",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = peft.PeftModel.from_pretrained(model_base, peft_model_name)
    else:
        model_name = "tiiuae/falcon-7b-instruct"
        print(f"Loading tokenizer: {model_name}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        print(f"Loading model: {model_name}", file=sys.stderr)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model_base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model_base.gradient_checkpointing_enable()
        model_base.enable_input_require_grads()
        peft_config = peft.LoraConfig(
            peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["self_attention.query_key_value"],
        )
        model = peft.get_peft_model(model_base, peft_config)
        model.print_trainable_parameters()

    opt = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0, betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss(reduction="none")

    model.train()
    batch_size = 4
    steps = round(len(data) / 2)

    pbar = tqdm(total=steps, desc="Training")
    for i in range(steps):
        batch = [next(data) for i in range(batch_size)]
        inputs = tokenizer(batch,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=4096).to("cuda")
        opt.zero_grad()
        outputs = model(inputs.input_ids[:, :-1], attention_mask=inputs.attention_mask[:, :-1], use_cache=False)
        losses = criterion(outputs.logits.transpose(-1, -2), inputs.input_ids[:, 1:])
        loss = torch.sum(losses * inputs.attention_mask[:, :-1]) / torch.sum(inputs.attention_mask[:, :-1])
        loss.backward()
        opt.step()
        pbar.update(1)
        pbar.set_description(f"Training (Train | Loss: {round(loss.item(),5)})")
    model.save_pretrained(continue_from if continue_from else "reward_models/default/",
                          safe_serialization=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset", help="The zipped tuning dataset for the evaluator.")
    args = parser.parse_args()
    data = ZippedConversationsDataset(args.dataset)
    lora_tune_evaluator(data)
