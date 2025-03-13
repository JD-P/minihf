from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForMaskedLM

parser = ArgumentParser()
parser.add_argument("tokenizer")
args = parser.parse_args()

AutoTokenizer.from_pretrained(args.tokenizer)
AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")
