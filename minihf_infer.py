import os
import json
import hashlib
from flask import Flask, request, jsonify, make_response
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.truncation_side = "left"
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", device_map="auto", load_in_8bit=True, torch_dtype=torch.float16)

reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
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
        x = self.linear1(x)
        x = self.gaussian1(x)
        x = self.linear2(x)
        return x

reward_head = RewardHead()
if os.path.exists("reward_heads/default.pkl"):
    print("Loading reward head...")
    # TODO: Safe tensors
    reward_head_state = torch.load("reward_heads/default.pkl")["model"]
    reward_head.load_state_dict(reward_head_state)
reward_head.to("cuda")
    
def score_output(text):
    with torch.cuda.amp.autocast():
        inputs = reward_tokenizer([text] * 1, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
        try:
            activations = reward_model_base(input_ids=inputs.input_ids)
        except:
            import pdb
            pdb.set_trace()
        penultimate_activation = activations["hidden_states"][-1]
        embedding = torch.mean(penultimate_activation, 1)
        score = reward_head(embedding)
        return float(score)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

app = Flask(__name__)

def generate_output(text, new_tokens):
    inputs = tokenizer([text] * 1, return_tensors="pt", truncation=True, max_length=(4096 - new_tokens)).to("cuda")
    input_length = inputs['input_ids'][0].shape[0]
    tokens = model.generate(
        inputs.input_ids,
        min_new_tokens=2,
        max_new_tokens=new_tokens,
        temperature=0.95,
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
        pad_token_id=0,
    )
    return [tokenizer.decode(t, skip_special_tokens=True) for t in tokens[:, inputs.input_ids.shape[1] :]]

class TreeNode:
    def __init__(self, parent, text):
        if parent == None:
            self.root = self
            self.depth = 0
        else:
            self.root = parent.root
            self.depth = parent.depth + 1
        self.parent = parent
        self.children = []
        self.text = text
        self.score = None

    def branch_text(self, include_root=False):
        branch_texts = [self.text]
        node = self
        while node.parent:
            node = node.parent
            branch_texts.append(node.text)
        if include_root:
            return " ".join(reversed(branch_texts))
        else:
            return " ".join(reversed(branch_texts[:-1]))

    def serialize_branch(self):
        branch_nodes = [{"depth": self.depth,
                         "text": self.text,
                         "score": self.score,
                         }]
        node = self
        while node.parent:
            node = node.parent
            serial_node = {"depth": node.depth,
                           "text": node.text,
                           "score": node.score,
                           }
            branch_nodes.append(serial_node)
        branch_nodes.reverse()
        return branch_nodes
    
    def add_child(self, text):
        child = TreeNode(self, text)
        self.children.append(child)
        return child
        
    def nodes(self):
        node_list = [self]
        for child in self.children:
            node_list.extend(child.nodes())
        return node_list
    
        
def weave_tree_search(tree: TreeNode, n: int, temperature: float = 1.0,
                      budget: int = 32, gen_tokens: int = 32,
                      expand_samples: int = 4, top_k: int = 4):
    """Tree search algorithm to find the top_k branches for a n token
    length completion given a fixed budget of gen_tokens length generations
    to explore for the whole tree."""
    # Make sure we've been given the root node
    assert tree.parent == None
    assert tree.root == tree
    max_depth = n / gen_tokens
    while budget: # Expand tree until budget is consumed
        # Selection - Select the node to expand
        eligible_nodes = [node for node in tree.nodes()
                          if (node.depth < max_depth) and not node.children]
        if not eligible_nodes: # Stop if all branches at max depth
            break
        node_scores = torch.tensor([node.score for node in eligible_nodes],
                                   dtype=torch.float32)
        node_scores = node_scores / temperature
        choice = torch.multinomial(torch.softmax(node_scores, 0), 1)
        chosen = eligible_nodes[choice]
        # Expansion - Expand the selected node
        chosen_branch_text = chosen.branch_text(include_root=True)
        for i in range(expand_samples):
            text = generate_output(chosen_branch_text, gen_tokens)[0]
            text_score = score_output(text)
            new_child = TreeNode(chosen, text)
            new_child.score = text_score
            chosen.children.append(new_child)
        budget -= expand_samples
    nodes = tree.nodes()
    nodes.sort(key=lambda x: x.score)
    return nodes[-top_k:]

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
        full_prompt = context + " " + prompt
        root_score = score_output(full_prompt)
        tree = TreeNode(None, full_prompt)
        tree.score = root_score
        new_tokens = int(params['new_tokens'])
        outs = weave_tree_search(tree=tree, n=new_tokens, temperature=1.0,
                                 budget=32, gen_tokens=32, expand_samples=4,
                                 top_k=4)
        batch = []
        for out in outs:
            out_text = out.branch_text()
            id_ = hashlib.md5(out_text.encode("UTF-8")).hexdigest()
            batch.append({"id":id_, "prompt": prompt,
                          "text":out_text, "nodes":out.serialize_branch()})
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
        inputs = tokenizer([text] * 1, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
        # TODO: Proper CORS
        response = jsonify(inputs['input_ids'][0].shape[0])
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

@app.route("/")
def index():
    return app.send_static_file("minihf.html")
