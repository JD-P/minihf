from argparse import ArgumentParser
import os
import re
import json
import time
import random
import hashlib
import zipfile
from contextlib import contextmanager
from functools import partial
from itertools import islice
from tqdm import tqdm
import torch
from weave import weave_tree_search, generate_outputs_vllm, evaluate_outputs_vllm
from weave import make_score_prompt_vllm, TreeNode


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def parse_constitution(cons):
    principles = {}
    raw_principles = re.split("==\[(.+)\]==", cons)[1:]
    principle_pairs = [i for i in batched(raw_principles, 2)]
    principle_pairs = [(i[0].strip(), i[1].strip()) for i in principle_pairs]
    principles["preamble"] = principle_pairs[0][1]
    principles["principles"] = []
    for pair in principle_pairs[1:]:
        principle = {}
        for parameter in pair[0].split(";"):
            try:
                name, value = parameter.split(":")
            except ValueError:
                raise ValueError(f"{pair} is missing a colon in a header value")
            principle[name.strip().lower()] = value.strip().lower()
        principle["body"] = pair[1].strip()
        principles["principles"].append(principle)
    return principles

def prepare_rubric(rubric_path, rubric_score_fn):
    with open(rubric_path) as infile:
        rubric = parse_constitution(infile.read())
        principle_weights = [float(principle["weight"]) for principle in rubric["principles"]]
        principle_weights = torch.tensor(principle_weights)
        principle_signs = []
        for principle in rubric["principles"]:
            answer = principle["answer"].lower()
            if answer not in {"yes", "no"}:
                raise ValueError("desired answer must be yes or no")
            principle_signs.append(1 if answer == "yes" else -1)
        principle_signs = torch.tensor(principle_signs)
    rubric_score_fns = []
    for principle in rubric["principles"]:
        evaluation_prompt = principle["body"].format(preamble=rubric["preamble"],
                                                     text="{text}")
        score_prompt_fn = partial(rubric_score_fn, evaluation_prompt)
        # FLAN evaluator LoRA suffix
        rubric_score_fns.append(partial(score_prompt_fn, "<|end|>"))
    return rubric_score_fns, principle_weights, principle_signs

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("prompt_path", help="Filepath to the prompt to start from.")
    parser.add_argument("rubric_path", help="Filepath to the grading rubric to use.")
    parser.add_argument("--generator", default=None)
    parser.add_argument("--evaluator", default="jdpressman/minihf_evaluator_mistral_7b_v0.1")
    args = parser.parse_args()

    generate_fn = partial(generate_outputs_vllm, args.generator)
    evaluate_fn = partial(evaluate_outputs_vllm, args.evaluator)

    with open(args.prompt_path) as infile:
        weave_prompt = infile.read()
    # TODO: Change weave to let me use q_weights and q_signs
    rubric_score_fns, q_weights, q_signs = prepare_rubric(args.rubric_path,
                                                          make_score_prompt_vllm)
    tree = TreeNode(weave_prompt)
    # Change name to avoid overwriting global baseline evaluate_fn partial
    score_fn = partial(evaluate_fn, rubric_score_fns)
    weave_param_defaults = {"weave_n_tokens":64, "weave_budget":144,
                            "weave_round_budget":24, "weave_n_expand":16,
                            "weave_beam_width":1, "weave_max_lookahead":3,
                            "weave_temperature":0.2}
    wp = weave_param_defaults
    # TODO: Let user specify these through a config file
    # for key in weave_param_defaults.keys():
    #    if key in params:
    #        try:
    #            wp[key] = int(params[key])
    #        except ValueError:
    #            wp[key] = float(params[key])
    #    else:
    #        wp[key] = weave_param_defaults[key]
    branches = []
    branches += weave_tree_search(tree=tree,
                                  generate_fn=partial(generate_fn,
                                                      n_tokens=wp["weave_n_tokens"]),
                                  evaluate_fn=score_fn,
                                  budget=wp["weave_budget"],
                                  round_budget=wp["weave_round_budget"],
                                  n_expand=wp["weave_n_expand"],
                                  beam_width=wp["weave_beam_width"],
                                  max_lookahead=wp["weave_max_lookahead"],
                                  temperature=wp["weave_temperature"])
    print(branches[-1].branch_text())
