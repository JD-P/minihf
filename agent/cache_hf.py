import os
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForMaskedLM

parser = ArgumentParser()
parser.add_argument("tokenizer")
args = parser.parse_args()

if os.path.exists("hf_token.txt"):
    with open("hf_token.txt") as infile:
        token = infile.read().strip()

AutoTokenizer.from_pretrained(args.tokenizer, token=token)
