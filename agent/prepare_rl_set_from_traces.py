import os
import random
import json
from argparse import ArgumentParser
from multiprocessing import Pool
from transformers import AutoTokenizer
from render_block import render_block

def init_worker(tokenizer_name):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def apply_error_penalty(block):
    if "outcome" not in block:
        return 0
    if block["outcome"]["error"] == "AssertionError":
        return 0.25
    elif block["outcome"]["error"]:
        return 0.5
    elif block["outcome"]["result"]:
        return 0
    
def process_block(task):
    trace, i = task
    block = trace[i]
    if "candidates" not in block:
        return None
    sample = {}
    try:
        completion_tokens = len(tokenizer(render_block(block), add_special_tokens=False)["input_ids"])
        tokens = completion_tokens
        context_size = 48000 - completion_tokens
        start = 0
        context_blocks = []
        for context_block in reversed(trace[start:i-1]):
            context_block_tokens = len(tokenizer(render_block(context_block), add_special_tokens=False)["input_ids"])
            tokens += context_block_tokens
            context_blocks.append(context_block)
            if tokens > 48000:
                break
        context = ""
        for event_block in reversed(context_blocks):
            context += render_block(event_block)
        if len(tokenizer(context, add_special_tokens=False)["input_ids"]) < context_size:
            print("Rejected")
            return None
        context = tokenizer.decode(
            tokenizer(context, add_special_tokens=False)["input_ids"][-context_size:]
        )
        sample["prompt"] = context
        sample["completions"] = []
        block2 = block.copy()
        block3 = block.copy()
        block2["body"] = block["candidates"][0][0]
        block2["score"] = block["candidates"][0][1]
        block3["body"] = block["candidates"][-1][0]
        block3["score"] = block["candidates"][-1][1]
        sample["completions"].append({
            "completion": render_block(block2),
            "reward": block2["score"]
        })
        sample["completions"].append({
            "completion": render_block(block3),
            "reward": block3["score"]
        })
        return sample
    except Exception as e:
        print(f"Error processing block {i}: {e}")
        return None

parser = ArgumentParser()
parser.add_argument("traces")
parser.add_argument("tokenizer")
parser.add_argument("--n", type=int, default=8, help="Number of processes to use")
args = parser.parse_args()

tasks = []
for filename in os.listdir(args.traces):
    filepath = os.path.join(args.traces, filename)
    if filepath.endswith(".json"):
        with open(filepath) as infile:
            trace = json.load(infile)
        for i, block in enumerate(trace):
            if "candidates" in block:
                tasks.append((trace, i))

with Pool(processes=args.n, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
    results = pool.map(process_block, tasks)

samples = [result for result in results if result]
random.shuffle(samples)

with open("rl_tuning_set.json", "w") as outfile:
    for sample in samples:
        outfile.write(json.dumps(sample) + "\n")
    outfile.flush()

