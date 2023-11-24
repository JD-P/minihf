#!/usr/bin/env python3

"""Fine-tunes a language model using reinforcement learning according to natural language
criteria."""

import argparse
from contextlib import contextmanager
from functools import partial
from itertools import islice
import math
import os
from pathlib import Path
import re
import sys

os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import accelerate
import dice_mc.torch as dice
import peft
import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print = tqdm.external_write_mode()(print)


def endless_range(start=0, step=1):
    i = start
    while True:
        yield i
        i += step


def constant_schedule(value=1.0):
    return lambda i: value


def exponential_warmup(beta, max_value=1.0):
    return lambda i: max_value * (1 - beta ** (i + 1))


def at_least_float32(tensor):
    dtype = torch.promote_types(tensor.dtype, torch.float32)
    return tensor.to(dtype)


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


def logsumexp_scaled(a, b, return_sign=False, dim=None, keepdim=False):
    """Compute log(sum(b * exp(a)))."""
    if dim is None:
        dim = tuple(range(a.ndim))

    a, b = torch.broadcast_tensors(a, b)
    a = torch.where(b != 0, a, float("-inf"))

    a_max = torch.amax(a, dim=dim, keepdim=True)
    a_max = torch.nan_to_num(a_max, 0.0, 0.0, 0.0)

    tmp = b * torch.exp(a - a_max)

    s = torch.sum(tmp, dim=dim, keepdim=keepdim)
    if return_sign:
        sgn = torch.sign(s)
        s *= sgn
    out = torch.log(s)

    if not keepdim:
        a_max = torch.squeeze(a_max, dim=dim)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def soft_maximum(values, weights=None, tau=1.0, dim=None, keepdim=False):
    if weights is None:
        weights = torch.ones_like(values)
    weights /= weights.sum(dim=dim, keepdim=True)
    return logsumexp_scaled(values / tau, weights, dim=dim, keepdim=keepdim) * tau


def soft_minimum(values, weights=None, tau=1.0, dim=None, keepdim=False):
    if weights is None:
        weights = torch.ones_like(values)
    weights /= weights.sum(dim=dim, keepdim=True)
    return -logsumexp_scaled(-values / tau, weights, dim=dim, keepdim=keepdim) * tau


def get_scores_from_logits(logits, pos_tokens, neg_tokens):
    logits = at_least_float32(logits[:, -1, :])
    logits = F.log_softmax(logits, dim=-1)
    pos = torch.logsumexp(logits[:, pos_tokens], dim=-1)
    neg = torch.logsumexp(logits[:, neg_tokens], dim=-1)
    rest = (1 - pos.exp() - neg.exp()).log()
    return torch.logaddexp(pos, rest - math.log(2)) - torch.logaddexp(neg, rest - math.log(2))


def find_token_for_string(tokenizer, prefix, s):
    tok_prefix = tokenizer(prefix).input_ids
    tok_prefix_s = tokenizer(prefix + s).input_ids
    if tok_prefix_s[: len(tok_prefix)] != tok_prefix:
        raise RuntimeError(f"{prefix!r} tokens are not a prefix of {prefix + s!r} tokens")
    return tok_prefix_s[len(tok_prefix)]


def find_tokens_for_strings(tokenizer, prefix, strings):
    return sorted(set([find_token_for_string(tokenizer, prefix, s) for s in strings]))


def make_get_scores(tokenizer, prefix):
    pos_tokens = find_tokens_for_strings(tokenizer, prefix, ["yes", "Yes", "YES"])
    neg_tokens = find_tokens_for_strings(tokenizer, prefix, ["no", "No", "NO"])
    return partial(get_scores_from_logits, pos_tokens=pos_tokens, neg_tokens=neg_tokens)


def kl_div_est(logp, logq):
    """Biased estimator of D_KL(P || Q) from log(p(x)) and log(q(x)), x sampled from p."""
    return torch.logaddexp(logp - logq, logq - logp) - math.log(2)


def inv_cumsum(x):
    """Inverse of cumulative sum."""
    out = x.clone()
    out[..., 1:] -= x[..., :-1]
    return out


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def parse_prompts(raw_prompts):
    return [prompt.strip() for prompt in raw_prompts.split("<|endprompt|>")]


def parse_constitution(cons):
    principles = {}
    raw_principles = re.split("==\[(.+)\]==", cons)[1:]
    principle_pairs = [i for i in batched(raw_principles, 2)]
    principle_pairs = [(i[0].strip(), i[1].strip()) for i in principle_pairs]
    principles["preamble"] = principle_pairs[0][1]
    principles["principles"] = []
    for pair in principle_pairs[1:]:
        principle = {}
        for parameter in pair[0].split(";"):
            try:
                name, value = parameter.split(":")
            except ValueError:
                raise ValueError(f"{pair} is missing a colon in a header value")
            principle[name.strip().lower()] = value.strip().lower()
        principle["body"] = pair[1].strip()
        principles["principles"].append(principle)
    return principles


def make_prompts_for_scoring(cons, texts):
    return [
        [principle["body"].format(**text, preamble=cons["preamble"]) + "<|end|>" for text in texts]
        for principle in cons["principles"]
    ]


def main():
    parser = argparse.ArgumentParser(
        __doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to lora to resume from")
    parser.add_argument("--batch-size", type=int, default=1, help="the batch size")
    parser.add_argument("--constitution", type=Path, required=True, help="the constitution to use")
    parser.add_argument(
        "--grad-accum-steps", type=int, default=4, help="the number of gradient accumulation steps"
    )
    parser.add_argument("--kl-weight", type=float, default=1.0, help="the KL weight")
    parser.add_argument("--length", type=int, default=64, help="the number of tokens to sample")
    parser.add_argument("--output-path", type=str, required=True, help="the output path")
    parser.add_argument("--prompts", type=Path, required=True, help="the prompts to use")
    parser.add_argument(
        "--save-every", type=int, default=250, help="the number of steps between saves"
    )
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device
    print0 = accelerator.on_local_main_process(print)

    prompts = parse_prompts(args.prompts.read_text())
    cons = parse_constitution(args.constitution.read_text())
    principle_weights = [float(principle["weight"]) for principle in cons["principles"]]
    principle_weights = torch.tensor(principle_weights, device=device)
    principle_signs = []
    for principle in cons["principles"]:
        answer = principle["answer"].lower()
        if answer not in {"yes", "no"}:
            raise ValueError("desired answer must be yes or no")
        principle_signs.append(1 if answer == "yes" else -1)
    principle_signs = torch.tensor(principle_signs, device=device)

    model_name = "openlm-research/open_llama_7b"
    evaluator_adapter_name = "RiversHaveWings/minihf_evaluator_openllama_7b"
    eval_split_batches = True

    tokenizer = AutoTokenizer.from_pretrained(evaluator_adapter_name)
    tokenizer.padding_side = "left"
    get_scores = make_get_scores(tokenizer, "<|end|>")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if accelerator.num_processes == 1 else {"": device},
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = peft.PeftModel.from_pretrained(model, evaluator_adapter_name, "evaluator")
    model.requires_grad_(False)

    peft_config = peft.LoraConfig(
        peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
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
            # "lm_head",
        ],
    )
    if args.resume:
        model.load_adapter(args.resume, "default", is_trainable=True) 
    else:
        model.add_adapter("default", peft_config)
        model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if accelerator.is_local_main_process:
        model.print_trainable_parameters()

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_len = inputs.input_ids.shape[1]

    opt = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98))
    sched = optim.lr_scheduler.LambdaLR(opt, constant_schedule(1.0))
    kl_sched = constant_schedule(args.kl_weight)

    model, opt, sched = accelerator.prepare(model, opt, sched)

    baseline = dice.EMABaseline(decay=0.98).to(device)
    baseline_kl = dice.EMABaseline(decay=0.98).to(device)

    accelerator.wait_for_everyone()

    for i in tqdm(endless_range(), disable=not accelerator.is_local_main_process):
        if i % 20 == 0:
            with set_adapter(accelerator.unwrap_model(model), "default"):
                demo_examples = min(4, len(prompts))
                demo_bs = math.ceil(demo_examples / accelerator.num_processes)
                start_idx = accelerator.local_process_index * demo_bs
                end_idx = start_idx + demo_bs
                outputs = accelerator.unwrap_model(model).generate(
                    inputs.input_ids[start_idx:end_idx],
                    attention_mask=inputs.attention_mask[start_idx:end_idx],
                    do_sample=True,
                    min_new_tokens=args.length,
                    max_new_tokens=args.length,
                    pad_token_id=tokenizer.eos_token_id,
                    top_k=0,
                )
            outputs = accelerator.gather(outputs)[:demo_examples]
            text = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
            print0("======")
            print0("\n===\n".join(text))
            print0("======")

        with accelerator.accumulate(model):
            with set_adapter(accelerator.unwrap_model(model), "default"):
                indices = torch.randint(0, len(prompts), (args.batch_size,), device=device)
                tokens = accelerator.unwrap_model(model).generate(
                    inputs.input_ids[indices],
                    attention_mask=inputs.attention_mask[indices],
                    do_sample=True,
                    min_new_tokens=args.length,
                    max_new_tokens=args.length,
                    pad_token_id=tokenizer.eos_token_id,
                    top_k=0,
                )
            attention_mask = torch.cat(
                [inputs.attention_mask[indices], torch.ones_like(tokens[:, input_len:])], dim=1
            )
            texts = [tokenizer.decode(t, skip_special_tokens=True) for t in tokens]

            with torch.no_grad(), set_adapter(accelerator.unwrap_model(model), None):
                outputs_orig = model(tokens, attention_mask=attention_mask)
            logits_orig = at_least_float32(outputs_orig.logits)
            logp_orig = dice.logp_categorical(
                logits_orig[:, input_len - 1 : -1], tokens[:, input_len:]
            )
            logp_orig_cumsum = torch.cumsum(logp_orig, dim=1)

            split_texts = [
                {"prompt": prompts[index], "response": text[len(prompts[index]) :]}
                for text, index in zip(texts, indices)
            ]
            eval_prompts = make_prompts_for_scoring(cons, split_texts)
            scores = []
            with torch.no_grad(), set_adapter(accelerator.unwrap_model(model), "evaluator"):
                for eval_prompt_batch in eval_prompts:
                    if eval_split_batches:
                        scores.append([])
                        for eval_prompt in eval_prompt_batch:
                            eval_inputs = tokenizer(eval_prompt, return_tensors="pt").to(device)
                            eval_outputs = model(eval_inputs.input_ids)
                            score = get_scores(eval_outputs.logits)
                            scores[-1].append(score)
                        scores[-1] = torch.cat(scores[-1])
                    else:
                        eval_inputs = tokenizer(
                            eval_prompt_batch, return_tensors="pt", padding=True
                        ).to(device)
                        eval_outputs = model(
                            eval_inputs.input_ids, attention_mask=eval_inputs.attention_mask
                        )
                        scores.append(get_scores(eval_outputs.logits))
            scores = torch.stack(scores, dim=1)
            scores = soft_minimum(
                scores * principle_signs[None], principle_weights[None], dim=1
            ).to(device)

            with set_adapter(accelerator.unwrap_model(model), "default"):
                outputs = model(tokens, attention_mask=attention_mask)
                logits = at_least_float32(outputs.logits)
                logp = dice.logp_categorical(logits[:, input_len - 1 : -1], tokens[:, input_len:])
                logp_sum = torch.sum(logp, dim=1)
                logp_cumsum = torch.cumsum(logp, dim=1)
                kls = inv_cumsum(kl_div_est(logp_cumsum.detach(), logp_orig_cumsum.detach()))

                losses_main = -F.logsigmoid(scores)
                losses_main = dice.cost_node(losses_main, [logp_sum])
                losses_main_global = accelerator.reduce(losses_main, "mean")
                losses_main += baseline(losses_main_global, [logp_sum])
                losses_kl = kls * kl_sched(i)
                losses_kl = dice.cost_node(losses_kl, [logp_cumsum])
                losses_kl_global = accelerator.reduce(losses_kl, "mean")
                losses_kl += baseline_kl(losses_kl_global, [logp_cumsum])
                loss_main = losses_main.mean()
                loss_kl = losses_kl.mean()
                loss = loss_main + loss_kl
                loss_global = accelerator.reduce(loss, "mean")
                accelerator.backward(loss)

            print0(
                f"step: {i}, loss: {loss_global.item():g}, main: {losses_main_global.mean().item():g}, kl: {losses_kl_global.mean().item():g}"
            )

            if accelerator.sync_gradients:
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.pow(2).sum().item()
                grad_norm **= 0.5
                print0(f"grad norm: {grad_norm:g}")

            opt.step()
            sched.step()
            opt.zero_grad()

        if accelerator.is_main_process and i > 0 and i % args.save_every == 0:
            print("Saving model...", file=sys.stderr)
            tokenizer.save_pretrained(args.output_path)
            accelerator.unwrap_model(model).save_pretrained(
                args.output_path, safe_serialization=True, selected_adapters=["default"]
            )


if __name__ == "__main__":
    main()
