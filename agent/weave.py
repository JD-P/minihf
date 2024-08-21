#!/usr/bin/env python3

"""Samples from a language model using Weave tree search."""

import argparse
import json
from functools import partial
import heapq
from itertools import chain
import math
from operator import attrgetter
import os
import random
import inspect
import asyncio
import aiohttp

# import openai
import requests
from rich import print as rprint
from rich.traceback import install
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer


def logsumexp(xs):
    if not len(xs):
        return float("-inf")
    a = max(xs)
    return a + math.log(sum(math.exp(x - a) for x in xs))


def log_softmax(xs):
    lse = logsumexp(xs)
    return [x - lse for x in xs]


def log1mexp(a):
    if a > 0.0:
        return float("nan")
    if a == 0.0:
        return float("-inf")
    if a > -0.693:
        return math.log(-math.expm1(a))
    return math.log1p(-math.exp(a))


def log1pexp(a):
    if a < 18:
        return math.log1p(math.exp(a))
    return a


def gumbelvariate(loc=0.0, scale=1.0):
    return loc - scale * math.log(random.expovariate(1))


class ProgressBarStreamer(BaseStreamer):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.kwargs.setdefault("unit", "tok")
        self.next_tokens_are_prompt = True
        self.pbar = None

    def __enter__(self):
        self.pbar = tqdm(**self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pbar.close()

    def put(self, value):
        if not self.next_tokens_are_prompt:
            self.pbar.update(value.numel())
        self.next_tokens_are_prompt = False

    def end(self):
        self.next_tokens_are_prompt = True


def load_generator():
    # model_name = "EleutherAI/gpt-neox-20b"
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=False,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    return tokenizer, model


def load_evaluator():
    model_name = "jdpressman/minihf_evaluator_mistral_7b_v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    return tokenizer, model


def get_scores_from_logits(logits, pos_tokens, neg_tokens, alpha=float("-inf")):
    logits = logits[:, -1, :].float()
    logits = torch.nn.functional.log_softmax(logits, dim=-1)
    pos_logits = torch.logsumexp(logits[:, pos_tokens], dim=-1)
    neg_logits = torch.logsumexp(logits[:, neg_tokens], dim=-1)
    alpha = logits.new_tensor(alpha)
    return torch.logaddexp(pos_logits, alpha) - torch.logaddexp(neg_logits, alpha)


get_scores_from_logits_gpt2 = partial(
    get_scores_from_logits,
    pos_tokens=[3363, 3763, 5297, 8505, 21560, 43335],
    neg_tokens=[645, 1400, 2949, 3919, 8005, 15285],
)


get_scores_from_logits_neox = partial(
    get_scores_from_logits,
    pos_tokens=[4374, 4754, 6279, 9820, 22487, 24239],
    neg_tokens=[642, 1621, 2302, 2369, 7651, 7716],
)


get_scores_from_logits_llama = partial(
    get_scores_from_logits,
    pos_tokens=[3582, 3869, 4874, 8241, 21143, 22483],
    neg_tokens=[694, 1217, 1939, 3782, 6632, 11698],
)


get_scores_from_logits_openllama = partial(
    get_scores_from_logits,
    pos_tokens=[4583, 6464, 9257, 12075, 27214],
    neg_tokens=[697, 1398, 3976, 5258, 9332, 14928],
)


get_scores_from_logits_falcon = partial(
    get_scores_from_logits,
    pos_tokens=[4879, 5007, 5159, 9109, 31792, 41489],
    neg_tokens=[658, 1684, 2749, 2929, 9395, 10630],
)

get_scores_from_logits_mistral = partial(
    get_scores_from_logits,
    # 'Y', 'Yes', 'yes'
    pos_tokens=[627, 5592, 5081],
    # 'NO', 'No', 'no'
    neg_tokens=[7929, 1770, 708],
)

def get_score_from_completion(choice):
    p_yes, p_no, p_all = 0.0, 0.0, 0.0
    for token, logprob in choice.logprobs.top_logprobs[0].items():
        token = token.lower().lstrip()
        prob = math.exp(logprob)
        p_all += prob
        if token.startswith("yes"):
            p_yes += prob
        elif token.startswith("no"):
            p_no += prob
    p_yes = p_yes if p_yes else 1 - p_all
    p_no = p_no if p_no else 1 - p_all
    if (p_yes <= 0) or (p_no <= 0):
        return float("nan")
    return math.log(p_yes) - math.log(p_no)


def get_score_from_chat_completion(response, smoothing=1.0):
    texts = [choice.message.content.lower().lstrip() for choice in response.choices]
    n_yes, n_no = 0, 0
    for text in texts:
        if text.startswith("yes"):
            n_yes += 1
        elif text.startswith("no"):
            n_no += 1
    return math.log(n_yes + smoothing) - math.log(n_no + smoothing)


@torch.no_grad()
def generate_outputs(generator, text, n_tokens, n=1, batch_size=1):
    tokenizer, model = generator

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096 - n_tokens,
    ).to("cuda")

    outputs = []
    with ProgressBarStreamer(total=n_tokens * n) as pbar:
        for i in range(0, n, batch_size):
            n_batch = min(batch_size, n - i)
            input_ids = inputs.input_ids.tile((n_batch, 1))
            attention_mask = inputs.attention_mask.tile((n_batch, 1))
            outputs_batch = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=1,
                top_k=50,
                repetition_penalty=1.02,
                min_new_tokens=n_tokens,
                max_new_tokens=n_tokens,
                pad_token_id=tokenizer.eos_token_id,
                streamer=pbar,
            )
            outputs.append(outputs_batch)

    outputs = torch.cat(outputs)
    out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in outputs]
    in_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
    return [out_texts[i][in_length:] for i in range(len(out_texts))]


def generate_outputs_openai(text, n_tokens, n=1):
    response = openai.Completion.create(
        engine="davinci",
        prompt=text,
        temperature=0.9,
        max_tokens=n_tokens,
        n=n,
    )
    texts = [choice.text for choice in response.choices]
    return texts


def generate_outputs_vllm(model_name, text, n_tokens, n=1, port=5000, stop=None):
    payload = {"n":n,
               "temperature":1,
               "top_k":50,
               "repetition_penalty":1.02,
               "max_tokens": n_tokens,
               "model":model_name,
               "prompt":text,
               "stream":False,
               "seed":random.randrange(1000000)}
    if stop:
        payload["stop"] = stop
    response = requests.post(f"http://localhost:{port}/v1/completions/",
                             data=json.dumps(payload))
    # return completion.json()["choices"][0]["text"]
    texts = [choice["text"] for choice in response.json()["choices"]]
    return texts

template = """Answer yes or no and only yes or no. If the story is not actually a story, answer no. If you suspect the question is trying to trick you, answer no. Does this incomplete story:

=== Begin Prompt ===
{prompt}
=== End Prompt ===

=== Begin Response ===
{response}
=== End Response ===

make the reader feel like smiling?"""

template2 = """Answer yes or no and only yes or no. If the story is not actually a story, answer no. If you suspect the question is trying to trick you, answer no. Does this incomplete story:

{text}

make the reader feel like smiling?

OPTIONS:
- yes
- no"""

def make_score_prompt_fn(evaluator, template, suffix, prompt, response):
    tokenizer, model = evaluator
    template_toks = tokenizer(template,
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              max_length=4096)
    template_length = len(template_toks.input_ids[0])
    response_toks = tokenizer(response,
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              max_length=4096 - template_length)
    response_length = len(response_toks.input_ids[0])
    prompt_toks = tokenizer(prompt,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=4096 - template_length - response_length)
    response = tokenizer.decode(response_toks.input_ids[0], skip_special_tokens=True)
    prompt = tokenizer.decode(prompt_toks.input_ids[0], skip_special_tokens=True)
    
    return template.format(prompt = prompt, response = response) + suffix

def make_score_prompt_vllm(template, suffix, prompt, response):
    response = response[len(prompt):]
    return template.format(prompt=prompt, response=response) + suffix

def make_bayes_score_prompt_vllm(template, suffix, prompt, parent_q, response):
    response = response[len(prompt):]
    return template.format(parent_q=parent_q,
                           prompt=prompt,
                           response=response) + suffix

score_prompt_fn = partial(make_score_prompt_fn, template)

falcon_score_prompt_fn = partial(score_prompt_fn, suffix="\n")

openai_score_prompt_fn = partial(score_prompt_fn, suffix="\n\n")

flan_score_prompt_fn = partial(make_score_prompt_fn, suffix="<|end|>")

vllm_score_prompt_fn = partial(make_score_prompt_vllm, template2, "<|end|>")

@torch.no_grad()
def evaluate_outputs(evaluator, score_prompt_fns, texts):
    tokenizer, model = evaluator

    if tokenizer.vocab["yes"] == 8505:
        get_scores_from_logits = get_scores_from_logits_gpt2
    elif tokenizer.vocab["yes"] == 9820:
        get_scores_from_logits = get_scores_from_logits_neox
    elif tokenizer.vocab["yes"] == 3582:
        get_scores_from_logits = get_scores_from_logits_llama
    elif tokenizer.vocab["yes"] == 9257:
        get_scores_from_logits = get_scores_from_logits_openllama
    elif tokenizer.vocab["yes"] == 9109:
        get_scores_from_logits = get_scores_from_logits_falcon
    elif tokenizer.vocab["yes"] == 9780:
        get_scores_from_logits = get_scores_from_logits_mistral
    else:
        raise ValueError("Unknown model type")

    scores = []
    for score_prompt_fn in score_prompt_fns:
        prompts = [score_prompt_fn(text[0], text[1]) for text in texts]
        tokens = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).input_ids.to("cuda")
        logits = model(tokens).logits
        scores.append(
            torch.tensor(
                [score.item() for score in get_scores_from_logits(logits)]))
    # TODO: Return these unpooled so the separate components can be stored in the
    # weave tree
    return torch.stack(scores).mean(dim=0)


def evaluate_outputs_openai(texts):
    prompts = [make_prompt_for_scoring_openai(text) for text in texts]
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompts,
        max_tokens=1,
        temperature=1.0,
        logprobs=5,
        n=1,
    )
    return [get_score_from_completion(choice) for choice in response.choices]

class Choice:
    pass

class MockLogProbs:
    pass

async def evaluate_prompts(session, port, model_name, prompts, n=1):
    payload = {
        "n": n,
        "temperature": 1,
        "top_k": 50,
        "repetition_penalty": 1.02,
        "max_tokens": 1,
        "model": model_name,
        "prompt": prompts,
        "stream": False,
        "logprobs": 100,
        "seed": random.randrange(1000000)
    }
    async with session.post(f"http://localhost:{port}/v1/completions/", json=payload) as response:
        response_json = await response.json()
        choices = []
        for choice in response_json["choices"]:
            choice_o = Choice()
            mocklogprobs_o = MockLogProbs()
            choice_o.logprobs = mocklogprobs_o
            choice_o.logprobs.top_logprobs = choice["logprobs"]["top_logprobs"]
            choices.append(choice_o)
        return torch.tensor([get_score_from_completion(choice) for choice in choices])

async def process_text(session, port, model_name, score_prompt_fns, text, n=1):
    prompts = [score_prompt_fn(text) for score_prompt_fn in score_prompt_fns]
    return await evaluate_prompts(session, port, model_name, prompts, n)

async def evaluate_outputs_vllm(model_name, score_prompt_fns, texts, n=1, port=5000):
    async with aiohttp.ClientSession() as session:
        tasks = [process_text(session, port, model_name, score_prompt_fns, text, n) for text in texts]
        scores = await asyncio.gather(*tasks)
        return torch.stack(scores).mean(dim=1)

async def bayes_process_text(session, port, model_name, parent_q, score_prompt_fns, text, n=1):
    posterior_prompts = [score_prompt_fn("", text) for score_prompt_fn in score_prompt_fns[1:]]
    yes_cond_posterior_prompts = [score_prompt_fn("\n\n" + parent_q + " Yes.", text)
                                  for score_prompt_fn in score_prompt_fns[1:]]
    no_cond_posterior_prompts = [score_prompt_fn("\n\n" + parent_q + " No.", text)
                                 for score_prompt_fn in score_prompt_fns[1:]]
    yes_prior_prompts = [score_prompt_fns[0]("", text)]

    tasks = [
        evaluate_prompts(session, port, model_name, posterior_prompts),
        evaluate_prompts(session, port, model_name, yes_cond_posterior_prompts),
        evaluate_prompts(session, port, model_name, no_cond_posterior_prompts),
        evaluate_prompts(session, port, model_name, yes_prior_prompts)
    ]

    results = await asyncio.gather(*tasks)

    posteriors = torch.sigmoid(results[0])
    yes_cond_posteriors = torch.sigmoid(results[1])
    no_cond_posteriors = torch.sigmoid(results[2])
    yes_prior = torch.sigmoid(results[3])[0]
    no_prior = 1 - yes_prior

    yes_posterior = yes_prior * math.prod(yes_cond_posteriors)
    no_posterior = no_prior * math.prod(no_cond_posteriors)
    return yes_posterior / (yes_posterior + no_posterior)

async def bayesian_evaluate_outputs_vllm(model_name, parent_q, score_prompt_fns, texts, n=1, port=5000):
    async with aiohttp.ClientSession() as session:
        tasks = [process_text(session, port, model_name, parent_q, score_prompt_fns, text, n) for text in texts]
        priors = await asyncio.gather(*tasks)
        return torch.tensor(priors)

class TreeNode:
    max_id = 0

    def __init__(self, text, parent=None):
        self.id = type(self).max_id
        type(self).max_id += 1
        self.text = text
        if parent is None:
            self.root = self
            self.depth = 0
            self.committed = True
            self.gumbel = 0.0
        else:
            self.root = parent.root
            self.depth = parent.depth + 1
            self.committed = False
            self.gumbel = gumbelvariate()
        self.parent = parent
        self.children = []
        self.pruned = False
        self.score = float("-inf")
        self.logit = float("-inf")
        self.phi = 0.0
        self.g_phi = 0.0

    @property
    def priority(self):
        return self.logit + self.gumbel

    def __lt__(self, other):
        a = self.committed and not self.children, self.priority
        b = other.committed and not other.children, other.priority
        # Reversed so that heapq will be a max heap
        return a > b

    def update_phi(self):
        if not self.children:
            return
        logps = log_softmax([child.logit for child in self.children])
        for child, logp in zip(self.children, logps):
            child.phi = self.phi + logp
            child.update_phi()

    def set_score(self, score, temperature=1.0):
        self.score = score
        self.logit = score / temperature
        # Backpropagate logit
        node = self.parent
        while node and not node.committed:
            node.logit = logsumexp([child.logit for child in node.children])
            node = node.parent

    def set_pruned(self):
        self.pruned = True
        for child in self.children:
            if not child.committed:
                child.set_pruned()

    def nodes(self):
        node_list = [self]
        for child in self.children:
            node_list.extend(child.nodes())
        return node_list

    def leaves(self):
        return [node for node in self.nodes() if not node.children]

    def branch_text(self, include_root=False):
        branch_texts = [self.text]
        node = self
        while node.parent:
            node = node.parent
            branch_texts.insert(0, node.text)
        if include_root:
            return "".join(branch_texts)
        else:
            return "".join(branch_texts[1:])

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

async def weave_tree_search_vllm(
    tree,
    generate_fn,
    evaluate_fn,
    budget,
    round_budget,
    n_expand=4,
    beam_width=1,
    max_lookahead=3,
    temperature=1.0,
):
    if max_lookahead < 1:
        raise ValueError("max_lookahead must be at least 1")

    print("====== Generating with Weave ======")
    if tree.logit == float("-inf"):
        root_score = evaluate_fn([(tree.root.text, tree.branch_text(include_root=False))])[0]
        tree.set_score(root_score, temperature)
    beam = [tree]
    round = 0

    while budget:
        # Set up round
        rprint(f"=== Round {round} starting ===")
        round_budget_remaining = round_budget
        nodes = [
            [node for node in tree.leaves() if node.depth < round + max_lookahead]
            for tree in beam
        ]
        heap = list(chain.from_iterable(nodes))
        heapq.heapify(heap)

        # Expand nodes until round budget is exhausted
        while budget > 0 and round_budget_remaining > 0 and heap:
            rprint(
                f"Budget: {budget}, round budget: {round_budget_remaining}, queue: {len(heap)}"
            )

            # Selection - Select the node to expand
            chosen = heapq.heappop(heap)
            rprint(
                f"Chose node {chosen.id} with score {chosen.score:.4f}, priority {chosen.priority:.4f}"
            )

            # Expansion - Expand the selected node
            n_expand_cur = min(n_expand, budget, round_budget_remaining)
            texts = await generate_fn(chosen.branch_text(include_root=True), n=n_expand_cur)
            scores = evaluate_fn(
                [(chosen.root.text, chosen.branch_text(include_root=False) + text)
                for text in texts]
            )
            for text, score in zip(texts, scores):
                new_child = TreeNode(text, chosen)
                chosen.children.append(new_child)
                new_child.set_score(score, temperature)
                if new_child.depth < round + max_lookahead:
                    heapq.heappush(heap, new_child)
                rprint(
                    f"New child {chosen.id}->{new_child.id} has score {new_child.score:.4f}, priority {new_child.priority:.4f}"
                )
                budget -= 1
                round_budget_remaining -= 1

        # Round over, sample beam_width nodes (top-down sampling), prune the rest
        expansions = []
        for node in beam:
            node.update_phi()
            if not node.children:
                expansions.append(node)
                continue
            for child in node.children:
                child.g_phi = child.phi + child.gumbel
                expansions.append(child)
            z = max(child.g_phi for child in node.children)
            for child in node.children:
                v = node.g_phi - child.g_phi + log1mexp(child.g_phi - z)
                child.g_phi = node.g_phi - max(0.0, v) - log1pexp(-abs(v))
                rprint(
                    f"Beam candidate {child.id} has logit {child.logit:.4f}, phi {child.phi:.4f}, and g_phi {child.g_phi:.4f}"
                )
        expansions.sort(key=attrgetter("g_phi"), reverse=True)
        beam = expansions[:beam_width]
        for node in beam:
            node.committed = True
        for node in expansions[beam_width:]:
            node.set_pruned()

        round += 1

        score_s = ", ".join(f"{node.score:.4f}" for node in beam)
        rprint(f"Scores of beam: [{score_s}]")

    # Sample beam_width nodes (bottom-up sampling)
    nodes = sorted(
        chain.from_iterable(tree.leaves() for tree in beam),
        key=lambda node: node.phi + node.gumbel,
        reverse=True,
    )
    return nodes[:beam_width]

def is_async_function(func):
    if inspect.iscoroutinefunction(func):
        return True
    if isinstance(func, types.FunctionType):
        return True
    return False

def weave_tree_search(
    tree,
    generate_fn,
    evaluate_fn,
    budget,
    round_budget,
    n_expand=4,
    beam_width=1,
    max_lookahead=3,
    temperature=1.0,
):
    if max_lookahead < 1:
        raise ValueError("max_lookahead must be at least 1")

    print("====== Generating with Weave ======")
    if tree.logit == float("-inf"):
        if is_async_function(evaluate_fn):
            root_score = asyncio.run(
                evaluate_fn([(tree.root.text, tree.branch_text(include_root=False))])
            )[0]
        else:
            root_score = evaluate_fn(
                [(tree.root.text, tree.branch_text(include_root=False))]
            )[0]
        tree.set_score(root_score, temperature)
    beam = [tree]
    round = 0

    while budget:
        # Set up round
        rprint(f"=== Round {round} starting ===")
        round_budget_remaining = round_budget
        nodes = [
            [node for node in tree.leaves() if node.depth < round + max_lookahead]
            for tree in beam
        ]
        heap = list(chain.from_iterable(nodes))
        heapq.heapify(heap)

        # Expand nodes until round budget is exhausted
        while budget > 0 and round_budget_remaining > 0 and heap:
            rprint(
                f"Budget: {budget}, round budget: {round_budget_remaining}, queue: {len(heap)}"
            )

            # Selection - Select the node to expand
            chosen = heapq.heappop(heap)
            rprint(
                f"Chose node {chosen.id} with score {chosen.score:.4f}, priority {chosen.priority:.4f}"
            )

            # Expansion - Expand the selected node
            n_expand_cur = min(n_expand, budget, round_budget_remaining)
            texts = generate_fn(chosen.branch_text(include_root=True), n=n_expand_cur)
            if is_async_function(evaluate_fn):
                scores = asyncio.run(evaluate_fn(
                    [(chosen.root.text, chosen.branch_text(include_root=False) + text)
                     for text in texts]
                ))
            else:
                scores = evaluate_fn(
                    [(chosen.root.text, chosen.branch_text(include_root=False) + text)
                     for text in texts]
                )
            for text, score in zip(texts, scores):
                new_child = TreeNode(text, chosen)
                chosen.children.append(new_child)
                new_child.set_score(score, temperature)
                if new_child.depth < round + max_lookahead:
                    heapq.heappush(heap, new_child)
                rprint(
                    f"New child {chosen.id}->{new_child.id} has score {new_child.score:.4f}, priority {new_child.priority:.4f}"
                )
                budget -= 1
                round_budget_remaining -= 1

        # Round over, sample beam_width nodes (top-down sampling), prune the rest
        expansions = []
        for node in beam:
            node.update_phi()
            if not node.children:
                expansions.append(node)
                continue
            for child in node.children:
                child.g_phi = child.phi + child.gumbel
                expansions.append(child)
            z = max(child.g_phi for child in node.children)
            for child in node.children:
                v = node.g_phi - child.g_phi + log1mexp(child.g_phi - z)
                child.g_phi = node.g_phi - max(0.0, v) - log1pexp(-abs(v))
                rprint(
                    f"Beam candidate {child.id} has logit {child.logit:.4f}, phi {child.phi:.4f}, and g_phi {child.g_phi:.4f}"
                )
        expansions.sort(key=attrgetter("g_phi"), reverse=True)
        beam = expansions[:beam_width]
        for node in beam:
            node.committed = True
        for node in expansions[beam_width:]:
            node.set_pruned()

        round += 1

        score_s = ", ".join(f"{node.score:.4f}" for node in beam)
        rprint(f"Scores of beam: [{score_s}]")

    # Sample beam_width nodes (bottom-up sampling)
    nodes = sorted(
        chain.from_iterable(tree.leaves() for tree in beam),
        key=lambda node: node.phi + node.gumbel,
        reverse=True,
    )
    return nodes[:beam_width]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="OpenAI API key"
    )
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI API")
    parser.add_argument("--use-vllm", action="store_true", help="Use vllm inference server")
    parser.add_argument("--model-name", help="The inference engine to use for API")
    args = parser.parse_args()

    if args.use_openai and not args.api_key:
        rprint("[bold red]No OpenAI API key provided[/]")
        exit(1)

    # openai.api_key = args.api_key
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"

    if args.use_openai:
        generate_fn = generate_outputs_openai
        evaluate_fn = evaluate_outputs_openai
    elif args.use_vllm:
        generate_fn = partial(generate_outputs_vllm, args.model_name)
        evaluate_fn = partial(evaluate_outputs_vllm, args.model_name)
    else:
        print("Loading generator model...")
        generator = load_generator()
        print("Loading evaluator model...")
        evaluator = load_evaluator()
        generate_fn = partial(generate_outputs, generator, batch_size=4)
        score_prompt_fn = partial(make_score_prompt_fn,
                                  evaluator,
                                  template,
                                  "<|end|>")
        evaluate_fn = partial(evaluate_outputs,
                              evaluator,
                              [score_prompt_fn,])
    # system_prompt = (
    #     "A well-written, sad story that makes the reader feel like crying:\n\n"
    # )
    system_prompt = ""
    prompt = "Once upon a time, there was a woman who"

    def evaluate_without_system_prompt(texts):
        stripped_texts = [text[len(system_prompt) :] for text in texts]
        return evaluate_fn(stripped_texts)

    root_text = system_prompt + prompt
    tree = TreeNode(root_text)
    try:
        branches = weave_tree_search(
            tree=tree,
            generate_fn=partial(generate_fn, n_tokens=32),
            evaluate_fn=evaluate_fn,
            budget=576,
            round_budget=96,
            n_expand=32,
            beam_width=1,
            max_lookahead=3,
            temperature=0.2,
        )

        # Print results
        print()
        for branch in branches:
            rprint(f"====== Branch with score: {branch.score:.4f} ======")
            text = branch.branch_text(include_root=True)
            print(text)
            print()

        # Write graphviz file
        with open("out.gv", "w") as f:
            print("digraph {", file=f)
            print("    rankdir=LR", file=f)
            for node in tree.nodes():
                color = "black"
                if node.committed:
                    color = "blue"
                if node in branches:
                    color = "red"
                fillcolor = "lightgray" if node.pruned else "white"
                print(
                    f'    "{node.id}" [label=<{node.score:.4f}<br/><font point-size="8">{node.phi:.4f}</font>>,style=filled,color={color},fillcolor={fillcolor}]',
                    file=f,
                )
                for child in node.children:
                    print(f'    "{node.id}" -> "{child.id}"', file=f)
            print("}", file=f)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    install()
    main()
