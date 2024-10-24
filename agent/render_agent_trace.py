import random
import json
from argparse import ArgumentParser
import torch
from render_block import render_block

parser = ArgumentParser()
parser.add_argument("trace", help="The JSON of the event blocks from the weave-agent.")
args = parser.parse_args()

with open(args.trace) as infile:
    events = json.load(infile)

context = ""
for event_block in events:
    context += render_block(event_block)

print(context)

