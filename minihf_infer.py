import os
import json
import random
import hashlib
import zipfile
from functools import partial
from flask import Flask, request, jsonify, make_response
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from weave import weave_tree_search, generate_outputs, evaluate_outputs
from weave import make_score_prompt_fn, TreeNode

def load_generator():
    # model_name = "EleutherAI/gpt-neox-20b"
    model_name = "EleutherAI/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        # load_in_4bit=False,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    return tokenizer, model

def load_evaluator():
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        # load_in_4bit=True,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    return tokenizer, model

generator = load_generator()
evaluator = load_evaluator()

generate_fn = partial(generate_outputs, generator, batch_size=4)
evaluate_fn = partial(evaluate_outputs, evaluator)

app = Flask(__name__)

@app.route("/generate", methods=['OPTIONS', 'POST'])
def generate():
    if request.method == 'OPTIONS':
        response = make_response()
        # TODO: Have the interface served by the server on GET request
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
    if request.method =='POST':
        params = request.get_json()
        prompt = params['prompt']
        context = params['context']
        full_prompt = context + " " + prompt
        new_tokens = int(params['new_tokens'])
        outs = generate_output(full_prompt, new_tokens)
        batch = []
        for out in outs:
          id_ = hashlib.md5(out.encode("UTF-8")).hexdigest()
          batch.append({"id":id_, "prompt": prompt, "text":out})
        # TODO: Proper CORS
        response = jsonify(batch)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

@app.route("/weave", methods=['OPTIONS', 'POST'])
def weave():
    if request.method == 'OPTIONS':
        response = make_response()
        # TODO: Have the interface served by the server on GET request
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
    if request.method =='POST':    
        params = request.get_json()
        prompt = params['prompt']
        context = params['context']
        evaluation_prompt = params['evaluationPrompt']
        full_prompt = context + " " + prompt
        tree = TreeNode(full_prompt)
        new_tokens = int(params['new_tokens'])
        score_prompt_fn = partial(make_score_prompt_fn, evaluation_prompt)
        # Falcon suffix
        score_prompt_fn = partial(score_prompt_fn, "\n")
        # Change name to avoid overwriting global baseline evaluate_fn partial
        score_fn = partial(evaluate_fn, score_prompt_fn)
        branches = weave_tree_search(tree=tree,
                                 generate_fn=partial(generate_fn, n_tokens=32),
                                 evaluate_fn=score_fn,
                                 budget=72,
                                 round_budget=24,
                                 n_expand=8,
                                 beam_width=1,
                                 max_lookahead=3,
                                 temperature=0.25)
        batch = []
        for branch in branches:
            branch_text = branch.branch_text()
            id_ = hashlib.md5(branch_text.encode("UTF-8")).hexdigest()
            batch.append({"id":id_,
                          "prompt": prompt,
                          "evaluationPrompt": evaluation_prompt,
                          "text":branch_text,
                          "nodes":branch.serialize_branch()})
        # TODO: Proper CORS
        response = jsonify(batch)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

@app.route("/check-tokens", methods=['OPTIONS', 'POST'])
def check_tokens():
    if request.method == 'OPTIONS':
        response = make_response()
        # TODO: Have the interface served by the server on GET request
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
    if request.method =='POST':    
        params = request.get_json()
        text = params['text']
        tokenizer, model = generator
        inputs = tokenizer([text] * 1, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
        # TODO: Proper CORS
        response = jsonify(inputs['input_ids'][0].shape[0])
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

class ZippedConversationsDataSet:
    def __init__(self, zip_file):
        self.training_items = []
        zip_ = zipfile.ZipFile(zip_file)
        for file_ in zip_.namelist():
            if file_.endswith("/"): # Skip directories
                continue
            if file_.startswith("__MACOSX"): # Mac OS X adds garbage to zips
                continue
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

def train_reward_head(zip_file):
    reward_head = RewardHead().to("cuda")
    reward_head.train()

    optimizer = torch.optim.Adam(reward_head.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    losses = []

    dataset = ZippedConversationsDataSet(zip_file)

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
        penultimate_activations = activations["hidden_states"][-1].float()
        embeddings = torch.sum(penultimate_activations * inputs.attention_mask[:, :, None], 1)
        embeddings = embeddings / torch.sum(inputs.attention_mask, 1)[:, None]
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
    # TODO: Safe tensors
    if not os.path.exists("reward_heads/"):
        os.mkdir("reward_heads/")
    torch.save({'model':reward_head.state_dict(), 'optimizer':optimizer.state_dict()},
               "reward_heads/default.pkl")

@app.route("/train-reward-model", methods=["OPTIONS", "POST"])
def train_reward_model():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
    if request.method =='POST':
        file_ = request.files['file']
        train_reward_head(file_)
        # TODO: Safe tensors
        reward_head_state = torch.load("reward_heads/default.pkl")["model"]
        reward_head.load_state_dict(reward_head_state)
        response = make_response("training complete")
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    
@app.route("/")
def index():
    return app.send_static_file("minihf.html")
