import os
import json
import time
import random
import hashlib
import zipfile
from functools import partial
from flask import Flask, request, jsonify, make_response
from tqdm import tqdm
import torch
import torch.nn as nn
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import BitsAndBytesConfig
from weave import weave_tree_search, generate_outputs, evaluate_outputs
from weave import make_score_prompt_fn, TreeNode
from lora_tune import lora_tune_evaluator
from dataset import ZippedConversationsDataset

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
    if os.path.exists("reward_models/default/"):
        peft_model_name = "./reward_models/default/"
        peft_config = peft.PeftConfig.from_pretrained(peft_model_name)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model_base = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = peft.PeftModel.from_pretrained(model_base, peft_model_name)
    else:
        model_name = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,
            # load_in_8bit=False,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    return tokenizer, model


def load_models():
    global generator
    generator = load_generator()
    global evaluator
    evaluator = load_evaluator()

    global generate_fn
    generate_fn = partial(generate_outputs, generator, batch_size=4)
    global evaluate_fn
    evaluate_fn = partial(evaluate_outputs, evaluator)

load_models()
    
app = Flask(__name__)

@app.route("/generate", methods=['OPTIONS', 'POST'])
def generate():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
    if request.method =='POST':
        params = request.get_json()
        prompt = params['prompt']
        if 'prompt_node' in params:
            prompt_node = params['prompt_node']
        else:
            prompt_node = False
        context = params['context']
        full_prompt = context + " " + prompt
        new_tokens = int(params['new_tokens'])
        outs = generate_fn(full_prompt, new_tokens)
        batch = []
        if prompt_node:
            timestamp = str(time.time())
            id_ = hashlib.md5((prompt + timestamp).encode("UTF-8")).hexdigest()
            batch.append({"id":id_,
                          "prompt":prompt,
                          "text":"",
                          "timestamp":timestamp,
                          "nodes":[]})
        for out in outs:
            timestamp = str(time.time())
            id_ = hashlib.md5(out.encode("UTF-8")).hexdigest()
            batch.append({"id":id_,
                          "prompt": prompt,
                          "text":out,
                          "timestamp":timestamp,
                          "nodes":[]})
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
        if 'prompt_node' in params:
            prompt_node = params['prompt_node']
        else:
            prompt_node = False
        evaluation_prompt = params['evaluationPrompt']
        full_prompt = context + " " + prompt
        tree = TreeNode(full_prompt)
        score_prompt_fn = partial(make_score_prompt_fn, evaluator)
        score_prompt_fn = partial(score_prompt_fn, evaluation_prompt)
        # Falcon suffix
        score_prompt_fn = partial(score_prompt_fn, "\n")
        # Change name to avoid overwriting global baseline evaluate_fn partial
        score_fn = partial(evaluate_fn, score_prompt_fn)
        weave_param_defaults = {"weave_n_tokens":32, "weave_budget":72,
                                "weave_round_budget":24, "weave_n_expand":8,
                                "weave_beam_width":1, "weave_max_lookahead":3,
                                "weave_temperature":0.25}
        wp = {}
        for key in weave_param_defaults.keys():
            if key in params:
                try:
                    wp[key] = int(params[key])
                except ValueError:
                    wp[key] = float(params[key])
            else:
                wp[key] = weave_param_defaults[key]
        branches = weave_tree_search(tree=tree,
                                     generate_fn=partial(generate_fn,
                                                         n_tokens=wp["weave_n_tokens"]),
                                     evaluate_fn=score_fn,
                                     budget=wp["weave_budget"],
                                     round_budget=wp["weave_round_budget"],
                                     n_expand=wp["weave_n_expand"],
                                     beam_width=wp["weave_beam_width"],
                                     max_lookahead=wp["weave_max_lookahead"],
                                     temperature=wp["weave_temperature"])
        batch = []
        if prompt_node:
            timestamp = str(time.time())
            id_ = hashlib.md5((prompt + timestamp).encode("UTF-8")).hexdigest()
            batch.append({"id":id_,
                          "prompt":prompt,
                          "evaluationPrompt":evaluation_prompt,
                          "text":"",
                          "timestamp":timestamp,
                          "nodes":[]})
        for branch in branches:
            branch_text = branch.branch_text()
            timestamp = str(time.time())
            id_ = hashlib.md5((branch_text + timestamp).encode("UTF-8")).hexdigest()
            batch.append({"id":id_,
                          "prompt": prompt,
                          "evaluationPrompt": evaluation_prompt,
                          "text":branch_text,
                          "timestamp":timestamp,
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
        # Deload models
        global generator
        del(generator)
        global evaluator
        del(evaluator)
        global generate_fn
        del(generate_fn)
        global evaluate_fn
        del(evaluate_fn)
        torch.cuda.empty_cache()
        data = ZippedConversationsDataset(file_)
        lora_tune_evaluator(data)
        torch.cuda.empty_cache()
        load_models()
        response = make_response("training complete")
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    
@app.route("/")
def index():
    return app.send_static_file("minihf.html")
