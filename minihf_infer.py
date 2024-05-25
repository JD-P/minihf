import os
import json
import time
import random
import hashlib
import zipfile
from contextlib import contextmanager
from functools import partial
from flask import Flask, request, jsonify, make_response
from tqdm import tqdm
import torch
import torch.nn as nn
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import BitsAndBytesConfig
from weave import weave_tree_search, generate_outputs, evaluate_outputs
from weave import make_score_prompt_fn, TreeNode
from lora_tune import lora_tune_evaluator
from dataset import ZippedConversationsDataset

@contextmanager
def set_adapter(model, adapter_name):
    old_adapter_name = model.active_adapter
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            print(adapter_name)
            yield model
        else:
            with model.disable_adapter():
                print("Reached here!")
                yield model
    finally:
        model.set_adapter(old_adapter_name)

def load_generator_evaluator():
    evaluator_adapter_name = "jdpressman/minihf_evaluator_mistral_7b_v0.1"
    generator_adapter_name = None
    peft_config = peft.PeftConfig.from_pretrained(evaluator_adapter_name)
    model_name = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(evaluator_adapter_name)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = peft.PeftModel.from_pretrained(model, evaluator_adapter_name, "evaluator")
    if generator_adapter_name:
        model.load_adapter(generator_adapter_name, "generator")
    peft_config = peft.LoraConfig(
        peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
    )
    return tokenizer, model

def load_models():
    global evaluator, evaluate_fn, generator, generate_fn
    tokenizer, model = load_generator_evaluator()
    evaluator = generator = (tokenizer, model)
    adapter_name = "generator" if "generator" in generator[1].peft_config else None
    generate_fn = set_adapter(generator[1], adapter_name)(partial(generate_outputs, generator, batch_size=1))
    evaluate_fn = set_adapter(evaluator[1], "evaluator")(partial(evaluate_outputs, evaluator))

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
        new_tokens = int(params['tokens_per_branch'])
        n_outputs = int(params['output_branches'])
        base_model_name = generator[1].active_peft_config.base_model_name_or_path
        try:
            adapter = params["adapter"]
        except KeyError:
            adapter = "generator" if "generator" in generator[1].peft_config else None
        if (adapter == "generator") or (adapter == None):
            gen_fn = generate_fn
        elif adapter == "evaluator":
            gen_fn = set_adapter(generator[1], "evaluator")(partial(generate_outputs, generator, batch_size=1))
        outs = gen_fn(prompt, new_tokens, n=n_outputs)
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
                          "base_model": base_model_name,
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
        # MiniHF evaluator LoRA suffix
        score_prompt_fn = partial(score_prompt_fn, "<|end|>")
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
    
@app.route("/")
def index():
    return app.send_static_file("minihf.html")
