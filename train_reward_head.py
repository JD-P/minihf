from argparse import ArgumentParser
from tqdm import tqdm
import os
import json
import zipfile
import random
import hashlib
from flask import Flask, request, jsonify, make_response
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

parser = ArgumentParser()
parser.add_argument("conversations_zip",
                    help="Path to the zipped conversation training set.")
args = parser.parse_args()

reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
reward_tokenizer.pad_token = reward_tokenizer.eos_token
reward_tokenizer.truncation_side = "left"
reward_model_base = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b-deduped",
                                             device_map="auto",
                                             load_in_8bit=True,
                                             torch_dtype=torch.float16)
reward_model_base.config.output_hidden_states = True

class RewardHead(nn.Module):
    def __init__(self):
        super(RewardHead, self).__init__()
        self.linear1 = nn.Linear(2048, 2048)
        self.gaussian1 = nn.GELU()
        self.linear2 = nn.Linear(2048, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.gaussian1(x)
        out = self.linear2(x)
        return out

reward_head = RewardHead().to("cuda")
reward_head.train()

class ZippedConversationsDataSet:
    def __init__(self, zip_file):
        self.training_items = []
        zip_ = zipfile.ZipFile(zip_file)
        for file_ in zip_.namelist():
            with zip_.open(file_) as infile:
                conversation = json.load(infile)
                for id_ in conversation["responseDict"]:
                    branch = conversation["responseDict"][id_]
                    if branch["rating"] == None: # Skip unrated entries
                        continue
                    text = ''
                    label = 1 if branch["rating"] else 0
                    for node in branch["nodes"]:
                        text += node["text"]
                        self.training_items.append((text, label))
        random.shuffle(self.training_items)

    def __len__(self):
        return len(self.training_items)
        
    def __next__(self):
        return random.sample(self.training_items, 1)[0]

optimizer = torch.optim.Adam(reward_head.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()
    
losses = []

# TODO: Safe tensors
outfile_name = os.path.split(args.conversations_zip)[1].replace("zip", "pkl")

dataset = ZippedConversationsDataSet(args.conversations_zip)

batch_size = 4
steps = round(len(dataset) / 2)

pbar = tqdm(total=steps, desc="Training")

for i in range(steps):
    batch = [next(dataset) for i in range(batch_size)]
    batch_texts = [x[0] for x in batch]
    batch_labels = torch.tensor([x[1] for x in batch]).to("cuda")
    inputs = reward_tokenizer(batch_texts,
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              max_length=4096).to("cuda")
    activations = reward_model_base(input_ids=inputs.input_ids)
    penultimate_activations = activations["hidden_states"][-1]
    embeddings = torch.mean(penultimate_activations, 1)
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        outs = reward_head(embeddings).squeeze(1)
    loss = criterion(outs, batch_labels.half())
    loss.backward()
    losses.append(loss.item())
    optimizer.step()
    pbar.update(1)
    avgloss = sum(losses) / len(losses)
    pbar.set_description(f"Training (Train | Loss: {round(avgloss,5)})")
    
torch.save({'model':reward_head.state_dict(), 'optimizer':optimizer.state_dict()},
           outfile_name)
