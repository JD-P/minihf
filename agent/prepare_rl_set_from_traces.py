import os
import random
import json
from argparse import ArgumentParser
from itertools import islice
from multiprocessing import Pool
from transformers import AutoTokenizer
from render_block import render_block

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def extract_chunks(tokenizer_name, traces_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    chunks = []
    for filename in os.listdir(traces_path):
        infile_path = os.path.join(traces_path, filename)
        with open(infile_path) as infile:
            blocks = json.load(infile)
        trace_text = ""
        for block in blocks:
            trace_text += render_block(block)
        chunk_tokens = list(batched(tokenizer(trace_text)["input_ids"], 48000))[:-1]
        chunks += [tokenizer.decode(tokens) for tokens in chunk_tokens]
    return chunks

def init_worker(tokenizer_name):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def process_trace_rewards(trace):
    index_map = {block["index"]: i for i, block in enumerate(trace) if "index" in block}
    
    # First pass - apply action outcome-based rewards
    for i, block in enumerate(trace):
        if "outcome" in block and block.get("type") == "action":
            # Find preceding block that needs adjustment
            preceding_types = ["orientation", "debug", "backtrack"]
            preceding_idx = None
            
            # Walk backwards to find nearest eligible block
            for j in range(i-1, -1, -1):
                if trace[j].get("type") in preceding_types:
                    preceding_idx = trace[j]["index"]
                    break
                if trace[j].get("type") == "orientation" and j != i-1:
                    break

            if preceding_idx is None:
                continue
            
            # Check action failure
            action_failed = False
            if i+1 < len(trace) and trace[i+1].get("type") == "error":
                action_failed = True

            # Calculate adjustment
            original_score = trace[index_map[preceding_idx]].get("score", 0.0)
            if action_failed:
                if original_score <= 2.0:
                    adjustment = -0.1
                elif 2 < original_score <= 3:
                    adjustment = -0.1 - (original_score - 2) * 0.1
                else:
                    adjustment = -0.2 - (original_score - 3) * 0.1
            else:
                if original_score < 2:
                    adjustment = 0.2
                elif 2 <= original_score <= 3:
                    adjustment = 0.1
                elif 3 < original_score <= 4:
                    adjustment = 0.01
                elif 4 < original_score <= 5:
                    adjustment = 0.001
                else:
                    adjustment = 0

            trace[index_map[preceding_idx]]["score"] += adjustment

    # Second pass - apply direct error penalties
    for block in trace:
        if "outcome" in block and block.get("score", 0) >= 0:
            error = block["outcome"].get("error")
            penalty = 0
            if error == "AssertionError":
                penalty = 0.25
            # Punish hallucinated methods harder
            elif error == "AttributeError":
                penalty = 1.0
            elif error:
                penalty = 0.5
            block["score"] = block["score"] - penalty

    # Apply rewards from evaluations
    for block in trace:
        if "reward" in block:
            block["score"] += block["reward"]["value"]
    
    return trace

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
        block2["body"] = block["candidates"][0][0]
        block2["score"] = block["candidates"][0][1]
        sample["completions"].append({
            "completion": render_block(block2),
            "reward": block2["score"]
        })
        sample["completions"].append({
            "completion": render_block(block),
            "reward": block["score"]
        })
        return sample
    except Exception as e:
        print(f"Error processing block {i}: {e}")
        return None
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("traces")
    parser.add_argument("tokenizer")
    parser.add_argument("--bad-traces")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    tasks = []
    for filename in os.listdir(args.traces):
        if filename.endswith(".json"):
            with open(os.path.join(args.traces, filename)) as f:
                trace = json.load(f)
            trace = process_trace_rewards(trace)
            # Create tasks after reward propagation
            for i, block in enumerate(trace):
                if "candidates" in block:
                    tasks.append((trace, i))

    with Pool(args.n, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        samples = list(filter(None, pool.map(process_block, tasks)))

    if args.bad_traces:
        good_chunks = extract_chunks(args.tokenizer, args.traces)
        bad_chunks = extract_chunks(args.tokenizer, args.bad_traces)
        random.shuffle(good_chunks)
        random.shuffle(bad_chunks)
        chunk_train_pairs = zip(good_chunks, bad_chunks)
        for pair in chunk_train_pairs:
            good, bad = pair
            sample = {}
            sample["prompt"] = ""
            sample["completions"] = []
            sample["completions"].append({
                "completion":good,
                "reward":10,
            })
            sample["completions"].append({
                "completion":bad,
                "reward":0,
            })
            samples.append(sample)
        
    random.shuffle(samples)

    with open("rl_tuning_set.json", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
