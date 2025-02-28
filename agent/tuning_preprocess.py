import sys
import json
import random
from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from functools import partial

parser = ArgumentParser()
parser.add_argument("training_format")
parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
parser.add_argument("--dataset", default="jdpressman/retroinstruct-agent-mix-v0.2")
parser.add_argument("--context-len", type=int, default=128000)
args = parser.parse_args()

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Dataset helper functions
def combine_flan(row):
    return f"<s> [INST] {row['inputs']} [/INST]{row['targets']}</s>"

def combine_dolly(row):
    return f"<s> [INST] {row['context']}\n\n{row['instruction']} [/INST]{row['response']}</s>"

def to_tokens(combine_fn, row):
    return tokenizer(combine_fn(row), add_special_tokens=False)

# Load dataset
print("### Loading datasets", file=sys.stderr)
dataset_1 = load_dataset("Open-Orca/FLAN")
dataset_2 = load_dataset("databricks/databricks-dolly-15k")
dataset_3 = load_dataset(args.dataset)

# Slice the first 150,000 items from dataset_1
dataset_1_sliced = dataset_1["train"].select(range(150000))

# Apply map to the sliced dataset_1 and the other datasets
dataset_1_sliced = dataset_1_sliced.map(partial(to_tokens, combine_flan))
dataset_2 = dataset_2["train"].map(partial(to_tokens, combine_dolly))
dataset_3 = dataset_3["train"].map(partial(to_tokens, combine_flan))

# Combine datasets
combined_dataset = concatenate_datasets([dataset_1_sliced, dataset_2, dataset_3])

# Shuffle the combined dataset
combined_dataset = combined_dataset.shuffle()

# Concatenate all rows into a single list of tokens
concatenated_tokens = []
for row in combined_dataset:
    concatenated_tokens.extend(row["input_ids"])

# Split into chunks of context_len tokens
chunk_size = args.context_len
chunks = []
for i in range(0, len(concatenated_tokens), chunk_size):
    chunks.append(concatenated_tokens[i:i + chunk_size])

# Shuffle the chunks
random.shuffle(chunks)

# Convert chunks back to text
chunked_texts = [tokenizer.decode(chunk) for chunk in chunks]
# assert len(tokenizer(chunked_texts[0], add_special_tokens=False)["input_ids"]) == 64000

train_val_test_ratios = [0.85, 0.1, 0.05]
train_len = int(train_val_test_ratios[0] * len(chunked_texts))
val_len = int(train_val_test_ratios[1] * len(chunked_texts))
test_len = int(train_val_test_ratios[2] * len(chunked_texts))

train = chunked_texts[:train_len]
val = chunked_texts[train_len:train_len+val_len]
test = chunked_texts[train_len+val_len:]

def write_dataset(filepath, data, _format):
    # Save to JSON lines file
    with open(filepath, 'w') as f:
        for text in data:
            if _format == "axolotl":
                f.write(json.dumps({"text": text}) + "\n")
            elif _format == "nemo":
                f.write(json.dumps({"input": "", "output": text}) + "\n")
        f.flush()

write_dataset("weave_train.jsonl", train, args.training_format)
write_dataset("weave_val.jsonl", val, args.training_format)
write_dataset("weave_test.jsonl", test, args.training_format)
        
print("### Dataset preprocessing complete", file=sys.stderr)
