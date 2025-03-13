import time
import re
import ast
import random
import os
import asyncio
import torch
from functools import partial
from rich import print as rprint
from weave import generate_outputs_vllm, evaluate_outputs_vllm
from weave import bayesian_evaluate_outputs_vllm
from weave import make_score_prompt_vllm, make_bayes_score_prompt_vllm
from weave import weave_tree_search_vllm, TreeNode
from render_block import render_block
from block_linters import lint_block

def make_simple_bayes_score_prompt(question: str):
    """Simplify the process of making a bayesian weave evaluator question prompt
    maker so that it's just a matter of passing a question for the weave-agent."""
    template = ("{response}\n"
                + "# Answer yes or no and only yes or no to the following\n"
                + "# question about the code block above.\n"
                + f"#q: {{parent_q}}\n# {question}")
    return partial(make_bayes_score_prompt_vllm, template, "", "")
    
def make_simple_score_prompt(question: str):
    """Simplify the process of making a weave evaluator question prompt maker so
    that it's just a matter of passing a question for the weave-agent."""
    template = ("<s> [INST] {response}\n"
                + "#q: If I flip a fair coin will it come up heads? No. (50%)\n"
                + "#q: X happens one in a hundred times. Did X happen? Yes. (1%)\n"
                + "#q: If I pick a US state at random will that state be Idaho? No. (98%)\n"
                + "#q: If I pick a book up from the thrift store will it be good according to Sturgeon's law? Yes. (10%)\n"
                + "# Answer yes or no and only yes or no to the following\n"
                + "# question about the code block above.\n"
                + f"#q: {question} [/INST]")
    return partial(make_score_prompt_vllm, template, "", "")

def mk_prompt(self, block_type, context, hint, retrieved_blocks=None):
    prompt = f'<s> [INST] {context}'
    if retrieved_blocks:
        prompt += f"# START RETRIEVED BLOCKS FOR BLOCK #{self.tree.current_block_index()}\n"    
        for i, block in enumerate(retrieved_blocks):
            block_text = f"# <retrieval{i}>\n" + block["render"][0].replace(
                block["type"][0],
                "recalled-" + block["type"][0],
                1
            ) + f"# </retrieval{i}>\n"
            block_text = "# " + block_text.replace("\n", "\n# ")
            prompt += "\n" + block_text
        prompt += f"\n# END RETRIEVED BLOCKS FOR BLOCK #{self.tree.current_block_index()}\n"
    prompt += f"\n# Write the next {block_type} block.\n"
    prompt += f"{hint}\n"
    # TODO: Fix this so it uses the rendered header from render_block ?
    time_remaining = self.end_time - time.time()
    prompt += f" [/INST]#subagent {self.name}\n"
    prompt += f"#startblock type: {block_type}\n"
    prompt += f"#time_remaining {time_remaining} seconds\n"

    # Narrow incidence of structurally wrong blocks by premising correct prefix
    if block_type == "orientation":
        prefix = '"""WEAVER [P: PLANNER], '
    elif block_type == "debug":
        prefix = '"""WEAVER [P: HYPOTHESIS], '
        with open("/app/error_stems.txt") as infile:
            stems = infile.readlines()
            stem = random.choice(stems)
            if "{timestamp}" in stem:
                error = self.tree.find_last_block_of_type("error")
                stem = stem.format(timestamp=error["timestamp"])
            prefix += stem
    elif block_type == "backtrack":
        prefix = '"""WEAVER [P: PLANNER], '
        with open("/app/backtrack_stems.txt") as infile:
            stems = infile.readlines()
            stem = random.choice(stems)
            prefix += stem
    elif block_type in {"expectation", "task-inference", "observation_inference"}:
        prefix = '"""'
    elif block_type in {"action", "evaluation"}:
        prefix = "def "
    else:
        prefix = ""
    prompt += prefix
    return prompt, prefix

async def generate_block_inner(self, block_type, context, eval_questions, weave_params, hint=""):
    def is_valid_syntax(code):
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            error_position = e.offset
            code_length = len(code)
            if code_length - error_position > 50:
                return False
            else:
                return True

    port = 5001

    prompt, prefix = mk_prompt(self, block_type, context, hint)

    if not os.path.exists("/app/weave-agent-logs/block-prompts/"):
        os.mkdir("/app/weave-agent-logs/block-prompts/")
    logpath = ("/app/weave-agent-logs/block-prompts/"
               + f"{block_type}_{self.tree.current_block_index()}.py")
    with open(logpath, "w") as outfile:
        outfile.write(prompt)
        outfile.flush()
    
    stopstrings = ["\n#q: ", "\n# q:", "#endblock", "#startblock"]
    generate_fn = partial(generate_outputs_vllm,
                          self.model_name,
                          port=port,
                          stop=stopstrings)
    score_prompt_fns = []
    # TODO: Use the full set of questions somehow?
    score_prompt_fns.append(make_simple_score_prompt(eval_questions[0]))
    async def evaluate_fn(texts, raw=False):
        score_fn = partial(evaluate_outputs_vllm,
                           self.model_name,
                           score_prompt_fns,
                           port=port)
        scores = await score_fn(texts)
        # Penalize syntax errors more than 50 characters from end of string
        syntax_penalties = torch.tensor([0 if is_valid_syntax(prefix + text[1]) else -2
                                         for text in texts])
        if block_type in {"task-inference"} and self.debugging:
            penalties = []
            for text in texts:
                penalty = False
                for string in [".completed", "complete"]:
                    if string in (prefix + text[1]):
                        penalty = True
                penalties.append(-2 if penalty else 0)
            completion_penalties = torch.tensor(penalties)
        else:
            completion_penalties = torch.zeros(len(scores))

        lint_penalties = torch.tensor([-1 * lint_block(block_type, prefix + text[1])
                                       for text in texts])
        if raw:
            return scores
        else:
            return scores + syntax_penalties + completion_penalties + lint_penalties
    tree = TreeNode(prompt)
    wp = weave_params
    # Rejection sample candidate for iterative retrieval
    query_candidates = await weave_tree_search_vllm(tree=tree,
                                                    generate_fn=partial(generate_fn,
                                                                        n_tokens=768),
                                                    evaluate_fn=evaluate_fn,
                                                    budget=4,
                                                    round_budget=4,
                                                    n_expand=4,
                                                    beam_width=1,
                                                    max_lookahead=1,
                                                    temperature=0.01)
    candidate = query_candidates[-1]

    retrieved_blocks = self.memory.search(prompt + candidate.branch_text(), limit=3)
    prompt, prefix = mk_prompt(self,
                               block_type,
                               context,
                               hint,
                               retrieved_blocks=retrieved_blocks)
    tree = TreeNode(prompt)
    # Rejection sample with retrieval
    branches = await weave_tree_search_vllm(tree=tree,
                                            generate_fn=partial(generate_fn,
                                                                n_tokens=768),
                                            evaluate_fn=evaluate_fn,
                                            budget=16,
                                            round_budget=16,
                                            n_expand=16,
                                            beam_width=1,
                                            max_lookahead=1,
                                            temperature=0.01)
    do_long = False
    if (branches[-1].score < 1):
        do_long = True
    try:
        program = branches[-1].branch_text()
        program = prefix + program.strip()
        compile(program, f"block_{self.tree.current_block_index()}", "exec")
    except Exception as e:
        do_long = True
        
    # If rejection sampling fails backtrack or do more rejection sampling
    if do_long:
        tree = TreeNode(prompt)
        branches = await weave_tree_search_vllm(tree=tree,
                                                generate_fn=partial(generate_fn,
                                                                    n_tokens=768),
                                                evaluate_fn=evaluate_fn,
                                                budget=128,
                                                round_budget=128,
                                                n_expand=128,
                                                beam_width=1,
                                                max_lookahead=1,
                                                temperature=0.01)        
        program = branches[-1].branch_text()
        # Check we finished writing the code block and extract first block
        stop_indices = []
        for stopstring in stopstrings:
            candidate = program.find(stopstring)
            if candidate > -1:
                stop_indices.append(candidate)
        if stop_indices:
            stop_index = min(stop_indices)
            program = prefix + program[:stop_index].strip()
        else:
            program = prefix + program.strip()
    block = {"type":block_type,
             "body":program,
             "q":eval_questions[0],
             "score":branches[-1].score.item()}
    try:
        compile(program, f"block_{self.tree.current_block_index()}", "exec")
    except Exception as e:
        block["score"] -= 2
        self.add_block(block)
        if len(self.tree.tokenizer(program)["input_ids"]) >= 768:
            raise ValueError("Length limit exceeded! Programs must be fewer than 768 tokens.")
        else:
            raise ValueError from e
    if block_type in {"orientation", "expectation", "debug", "backtrack"}:
        try:
            block["body"] = extract_first_string_literal(program)
        except AttributeError:
            raise ValueError("No string literal found in generated block.")
    if block_type in {"action", "evaluation"}:
        callback, registration = extract_function_and_add_action_or_evaluation(
            program,
            f"add_{block_type}"
        )
        block["body"] = callback + "\n\n" + registration
    raw_score = await evaluate_fn(
        [prompt[:len(prompt) - len(prefix)] + block["body"],],
        raw=True
    )
    block["raw_score"] = raw_score[0].item()
    self.add_block(block)
    rprint(f"Finished writing block #[cyan]{self.tree.current_block_index()-1}[/cyan] of type [cyan]{block_type}[/cyan]")
    print(block["body"])
    return block

def extract_first_string_literal(code):
    class StringLiteralVisitor(ast.NodeVisitor):
        def __init__(self):
            self.first_string_literal = None

        def visit_Str(self, node):
            if self.first_string_literal is None:
                self.first_string_literal = node.s

    # Parse the code into an AST
    tree = ast.parse(code)

    # Visit the nodes in the AST
    visitor = StringLiteralVisitor()
    visitor.visit(tree)

    return '"""' + visitor.first_string_literal.strip() + '"""'

def extract_function_and_add_action_or_evaluation(code, slot_name):
    class FunctionAndAddActionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.function_def = None
            self.add_action_call = None

        def visit_FunctionDef(self, node):
            if self.function_def is None:
                self.function_def = node

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == slot_name:
                self.add_action_call = node.value

    # Parse the code into an AST
    tree = ast.parse(code)

    # Visit the nodes in the AST
    visitor = FunctionAndAddActionVisitor()
    visitor.visit(tree)

    # Extract the function definition and the add_action call
    function_code = ast.unparse(visitor.function_def) if visitor.function_def else ""
    add_action_code = ast.unparse(visitor.add_action_call) if visitor.add_action_call else ""

    return function_code, add_action_code
