#!/usr/bin/env python3
# Written by DeepSeek and therefore public domain
"""
JSON directories of weave agent traces to Gzipped JSONL Shards with Category Wrapping
Combine JSON files from multiple directories into gzipped JSON lines shards,
wrapping each file's content with category information from the parent directory.
"""

import os
import json
import gzip
import math
import random
from pathlib import Path
import argparse
from tqdm import tqdm

def find_json_files(directories):
    """Find all JSON files in the given directories and return with their categories."""
    json_files_with_category = []
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory {directory} does not exist, skipping")
            continue
        
        # Find all JSON files and get their immediate parent directory as category
        for json_file in dir_path.rglob("*.json"):
            # Get the immediate parent directory name as category
            category = json_file.parent.name
            json_files_with_category.append((json_file, category))
    
    return json_files_with_category

def read_and_wrap_json_file(file_path, category):
    """Read a JSON file containing a list of dictionaries and wrap it with category."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        # Validate that content is a list of dictionaries
        if not isinstance(content, list):
            print(f"Warning: {file_path} does not contain a list, skipping")
            return None
        
        if content and not all(isinstance(item, dict) for item in content):
            print(f"Warning: {file_path} does not contain a list of dictionaries, skipping")
            return None
        
        # Wrap the content with category
        wrapped_content = {
            "trace": content,
            "category": category
        }
        
        return wrapped_content
        
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error processing {file_path}: {e}")
        return None

def estimate_total_size(json_files_with_category):
    """Estimate total uncompressed size by reading all files."""
    print("Estimating total size...")
    total_size = 0
    valid_files = []
    
    for file_path, category in tqdm(json_files_with_category, desc="Estimating sizes"):
        wrapped_content = read_and_wrap_json_file(file_path, category)
        if wrapped_content is None:
            continue
        
        # Estimate size using JSON string length
        json_string = json.dumps(wrapped_content, ensure_ascii=False)
        content_size = len(json_string.encode('utf-8'))
        total_size += content_size
        valid_files.append((file_path, category, wrapped_content, content_size))
    
    return valid_files, total_size

def process_directories_to_shards(directories, output_dir, shard_size=250*(10**6), shuffle=True):
    """Process JSON files from directories into gzipped JSONL shards with category wrapping."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files with their categories
    print("Finding JSON files and extracting categories...")
    json_files_with_category = find_json_files(directories)
    
    if not json_files_with_category:
        print("No JSON files found in the specified directories")
        return
    
    print(f"Found {len(json_files_with_category)} JSON files")
    
    # Estimate total size and get valid files with their sizes
    valid_files, total_size = estimate_total_size(json_files_with_category)
    
    if not valid_files:
        print("No valid JSON files found")
        return
    
    # Show categories found
    categories = set(category for _, category, _, _ in valid_files)
    print(f"Categories found: {', '.join(sorted(categories))}")
    print(f"Total uncompressed size: {total_size:,} bytes")
    
    # Calculate number of shards based on actual total size
    num_shards = max(1, math.ceil(total_size / shard_size))
    print(f"Creating {num_shards} shards of ~{shard_size:,} bytes each")
    
    random.shuffle(valid_files)
    
    # Process files into shards
    shard_index = 0
    current_shard_items = []
    current_shard_size = 0
    
    progress = tqdm(total=len(valid_files), desc="Processing files")
    
    for file_path, category, wrapped_content, content_size in valid_files:
        # If adding this file would exceed shard size and we already have some items, start new shard
        if current_shard_items and (current_shard_size + content_size > shard_size):
            # Write current shard
            write_shard(output_dir, shard_index, num_shards, current_shard_items)
            shard_index += 1
            current_shard_items = []
            current_shard_size = 0
        
        current_shard_items.append(wrapped_content)
        current_shard_size += content_size
        progress.update(1)
    
    # Write final shard if there are remaining items
    if current_shard_items:
        write_shard(output_dir, shard_index, num_shards, current_shard_items)
    
    progress.close()
    print(f"Successfully created {shard_index + 1} shards in {output_dir}")
    
    # Show actual shard sizes
    print("\nShard sizes:")
    for i in range(shard_index + 1):
        filename = f"train-{i:05d}-of-{num_shards:05d}.jsonl.gz"
        filepath = output_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  {filename}: {size:,} bytes (compressed)")

def write_shard(output_dir, shard_index, num_shards, contents):
    """Write a single gzipped JSONL shard."""
    filename = f"train-{shard_index:05d}-of-{num_shards:05d}.jsonl.gz"
    filepath = output_dir / filename
    
    with gzip.open(filepath, 'wt', encoding='utf-8') as outfile:
        for item in contents:
            json_line = json.dumps(item, ensure_ascii=False)
            outfile.write(json_line + "\n")
    
    print(f"Created shard: {filename} with {len(contents)} items")

def validate_output(output_dir, num_samples=5):
    """Validate the output by reading a few samples from each shard."""
    output_dir = Path(output_dir)
    shard_files = list(output_dir.glob("*.jsonl.gz"))
    
    if not shard_files:
        print("No shard files found for validation")
        return
    
    print("\nValidating output structure...")
    for shard_file in shard_files[:2]:  # Check first 2 shards
        print(f"\nChecking {shard_file.name}:")
        try:
            with gzip.open(shard_file, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    data = json.loads(line)
                    print(f"  Sample {i+1}: category='{data.get('category')}', trace_length={len(data.get('trace', []))}")
        except Exception as e:
            print(f"  Error reading {shard_file}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Combine JSON files from directories into gzipped JSON lines shards with category wrapping"
    )
    parser.add_argument(
        "directories", 
        nargs="+",
        help="Directories containing JSON files to process"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default="shards",
        help="Output directory for shards (default: shards)"
    )
    parser.add_argument(
        "--shard-size", 
        type=int, 
        default=250 * (10 ** 6),  # 250MB default
        help="Target shard size in bytes (default: 250000000)"
    )
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate output by reading samples from shards"
    )
    
    args = parser.parse_args()

    process_directories_to_shards(
        directories=args.directories,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
    )
    
    if args.validate:
        validate_output(args.output_dir)

if __name__ == "__main__":
    main()
