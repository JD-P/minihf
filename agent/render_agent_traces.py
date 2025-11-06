import random
import json
import gzip
import os
from argparse import ArgumentParser
import torch
from datasets import load_dataset
from render_block import render_block

parser = ArgumentParser()
parser.add_argument("trace", help="The JSON file or gzipped JSONL file containing the event blocks.")
parser.add_argument("--num-samples", type=int, default=1, help="Number of random samples to display (default: 1)")
parser.add_argument("--show-category", action="store_true", help="Show the category of each trace")
parser.add_argument("--all", action="store_true", help="Process all traces in the file (ignores --num-samples)")
parser.add_argument("--index", type=int, help="Process a specific trace by index (0-based)")
args = parser.parse_args()

def read_traces(file_path):
    """Read traces from either a regular JSON file or gzipped JSONL file."""
    traces = []
    categories = []
    
    file_ext = os.path.splitext(file_path)[1].lower()

    
    if file_ext == '.gz':
        # Gzipped JSONL format (new format)
        with gzip.open(file_path, 'rt') as infile:
            for line in infile:
                data = json.loads(line.strip())
                traces.append(json.loads(data.get("trace", [])))
                categories.append(data.get("category", "unknown"))
    elif file_ext == '.json':
        # Regular JSON format (old format - for backward compatibility)
        with open(file_path) as infile:
            data = json.load(infile)
            # Check if it's the new wrapped format or old direct format
            if isinstance(data, dict) and "trace" in data:
                # New format but in a single JSON file
                traces.append(json.loads(data.get("trace", [])))
                categories.append(data.get("category", "unknown"))
            elif isinstance(data, list):
                # Old format - direct list of events
                traces.append(data)
                categories.append("legacy")
            else:
                raise ValueError("Unsupported JSON format")
    else:
        dataset = load_dataset(file_path)
        for row in dataset["train"]:
            traces.append(json.loads(row["trace"]))
            categories.append(row["category"]) # good/bad labels for RL or similar
            
    
    return traces, categories

# Read traces from the file
try:
    traces, categories = read_traces(args.trace)
except Exception as e:
    print(f"Error reading file {args.trace}: {e}")
    exit(1)

if not traces:
    print("No traces found in the file.")
    exit(1)

print(f"Loaded {len(traces)} traces from {args.trace}")

# Select which traces to process
if args.all:
    # Process all traces
    indices = range(len(traces))
elif args.index is not None:
    # Process specific index
    if args.index < 0 or args.index >= len(traces):
        print(f"Error: Index {args.index} is out of range. File contains {len(traces)} traces.")
        exit(1)
    indices = [args.index]
else:
    # Select random samples
    num_samples = min(args.num_samples, len(traces))
    indices = random.sample(range(len(traces)), num_samples)

# Process selected traces
for i, trace_index in enumerate(indices):
    trace_events = traces[trace_index]
    category = categories[trace_index]
    
    print(f"\n{'='*80}")
    if args.show_category:
        print(f"TRACE {i+1} (Index: {trace_index}, Category: {category}):")
    else:
        print(f"TRACE {i+1} (Index: {trace_index}):")
    print(f"{'='*80}")
    
    context = ""
    for event_block in trace_events:
        context += render_block(event_block)
    
    print(context)
    
    if not args.all and i < len(indices) - 1:
        print(f"\n{'-'*80}")
        print("NEXT SAMPLE:")
        print(f"{'-'*80}")

print(f"\nDisplayed {len(indices)} trace(s) out of {len(traces)} total.")
