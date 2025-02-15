import time
import re
import ast
import random
import asyncio
import torch
from functools import partial
from rich import print as rprint
from weave import generate_outputs_vllm, evaluate_outputs_vllm
from weave import bayesian_evaluate_outputs_vllm
from weave import make_score_prompt_vllm, make_bayes_score_prompt_vllm
from weave import weave_tree_search, TreeNode
from render_block import render_block
from block_linters import lint_block

def make_simple_bayes_score_prompt(question: str):
    """Simplify the process of making a bayesian weave evaluator question prompt
    maker so that it's just a matter of passing a question for the weave-agent."""
    template = ("{response}\n\n"
                + "# Answer yes or no and only yes or no to the following.\n"
                + "# question about the incomplete code block above.\n"
                + "# Keep in mind the following question is being asked as part\n"
                + "# of a Monte Carlo Tree Search so the above is usually a work in progress.\n"
                + "# You're really being asked something like *will this trajectory*\n"
                + "# eventually have quality X or satisfy predicate Y?\n"
                + f"#q: {{parent_q}}\n# {question}")
    return partial(make_bayes_score_prompt_vllm, template, "", "")
    
def make_simple_score_prompt(question: str):
    """Simplify the process of making a weave evaluator question prompt maker so
    that it's just a matter of passing a question for the weave-agent."""
    template = ("<s> [INST] {response}\n\n"
                + "#q: If I flip a fair coin will it come up heads? No. (50%)\n"
                + "#q: X happens one in a hundred times. Did X happen? Yes. (1%)\n"
                + "#q: If I pick a US state at random will that state be Idaho? No. (98%)\n"
                + "#q: If I pick a book up from the thrift store will it be good according to Sturgeon's law? Yes. (10%)\n"
                + "# Answer yes or no and only yes or no to the following.\n"
                + "# question about the incomplete code block above.\n"
                + "# Keep in mind the following question is being asked as part\n"
                + "# of a Monte Carlo Tree Search so the above is usually a work in progress.\n"
                + "# You're really being asked something like *will this trajectory*\n"
                + "# eventually have quality X or satisfy predicate Y?\n"
                + f"#q: {question} [/INST]")
    return partial(make_score_prompt_vllm, template, "", "")

def generate_block_inner(self, block_type, context, eval_questions, weave_params, hint=""):
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

    bm25_prompt =  f'<s> [INST] {context} [/INST]#startblock type: {block_type}\n'
    bm25_prompt += "#timestamp {time.time()}\n"
    bm25_prompt += "# I need to write a Tantivy BM 25 query to retrieve relevant blocks below.\n"
    bm25_prompt += "# After the tag bm25_query I will write a series of words "
    bm25_prompt += "# summarizing the situation. I will also search for words and "
    bm25_prompt += "# phrases related to the problem I need to solve in the next block."
    bm25_prompt += "# I put a + in front of a word +like +so to denote that I want "
    bm25_prompt += "# to search for it and I put a - in front of a word to denote "
    bm25_prompt += "# that I want to avoid it -like -so. Write a long series of "
    bm25_prompt += "# relevant search words below, try to get at least a dozen:\n"
    if self.tree.current_block_index() < 50:
        bm25_prompt += f"#bm25_query type:'{block_type}' "
    else:
        bm25_prompt += "#bm25_query "

    port = 5001
    # TODO: Rejection sample this?
    query_candidates = generate_outputs_vllm(self.model_name,
                                             bm25_prompt,
                                             256,
                                             port=port,
                                             n=8,
                                             stop=["\n",])
    bm25_query = None
    for candidate in query_candidates:
        # Prevent duplicate entries taking over through autoregressive reinforcement
        candidate = " ".join([term for term in set(candidate.split(" "))])
        try:
            self.bm25_index.parse_query(candidate, ["render", "description"])
            bm25_query = candidate
            break
        except ValueError:
            continue
    if bm25_query:
        bm25_query = f"type:'{block_type}' " + bm25_query

        searcher = self.bm25_index.searcher()
        query = self.bm25_index.parse_query(bm25_query, ["render", "description"])
        results = searcher.search(query, limit=25).hits
        retrieved_blocks = [searcher.doc(result[1]) for result in results
                            if searcher.doc(result[1])["score"][0] >= 2]
        retrieved_blocks = sorted(retrieved_blocks,
                                  key=lambda block: block["score"][0],
                                  reverse=True)[:3]
    else:
        retrieved_blocks = None

    prompt = f'<s> [INST] {context}'
    if retrieved_blocks:
        prompt += f"# START RETRIEVED BLOCKS FOR BLOCK #{self.tree.current_block_index()}\n"    
        for block in retrieved_blocks:
            block_text = block["render"][0].replace(
                block["type"][0],
                "recalled-" + block["type"][0]
            )
            block_text = "# " + block_text.replace("\n", "\n# ")
            prompt += "\n" + block_text
        prompt += f"\n# END RETRIEVED BLOCKS FOR BLOCK #{self.tree.current_block_index()}\n"
    prompt += f"\n# Write the next {block_type} block."
    # TODO: Fix this so it uses the rendered header from render_block ?
    time_remaining = self.end_time - time.time()
    prompt += f" [/INST]#subagent {self.name}\n"
    prompt += f"#startblock type: {block_type}\n"
    prompt += f"#time_remaining {time_remaining} seconds\n"
    prompt += f"{hint}\n"

    # Narrow incidence of structurally wrong blocks by premising correct prefix
    if block_type == "orientation":
        prefix = '"""WEAVER [P: EXPECTATION], '
    elif block_type == "debug":
        prefix = '"""WEAVER [P: HYPOTHESIS], '
        with open("/app/error_stems.txt") as infile:
            stems = infile.readlines()
            stem = random.choice(stems)
            if "{timestamp}" in stem:
                error = self.tree.find_last_block_of_type("error")
                stem = stem.format(timestamp=error["timestamp"])
            prefix += stem
    elif block_type in {"expectation", "task-inference", "observation_inference"}:
        prefix = '"""'
    elif block_type in {"action", "evaluation"}:
        prefix = "def "
    else:
        prefix = ""
    #if block_type in {"orientation"} and self.debugging:
    #    with open("/app/error_stems.txt") as infile:
    #        error_stem = random.choice(infile.readlines())
    #        prefix += error_stem.strip().format(stage=self.failure_stage) + " "
    prompt += prefix
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
    # First try a simple rejection sampling
    branches = weave_tree_search(tree=tree,
                                 generate_fn=partial(generate_fn,
                                                     n_tokens=768),
                                 evaluate_fn=evaluate_fn,
                                 budget=32,
                                 round_budget=32,
                                 n_expand=32,
                                 beam_width=1,
                                 max_lookahead=1,
                                 temperature=0.01)
    do_long = False
    if (branches[-1].score < 2.5):
        do_long = True
    try:
        program = branches[-1].branch_text()
        program = prefix + program.strip()
        compile(program, f"block_{self.tree.current_block_index()}", "exec")
    except Exception as e:
        do_long = True
    # If rejection sampling fails do full search
    if do_long:
        tree = TreeNode(prompt)
        branches = weave_tree_search(tree=tree,
                                     generate_fn=partial(generate_fn,
                                                         n_tokens=wp["weave_n_tokens"]),
                                     evaluate_fn=evaluate_fn,
                                     budget=wp["weave_budget"],
                                     round_budget=wp["weave_round_budget"],
                                     n_expand=wp["weave_n_expand"],
                                     beam_width=wp["weave_beam_width"],
                                     max_lookahead=wp["weave_max_lookahead"],
                                     temperature=wp["weave_temperature"]) 
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
    if bm25_query:
        block["bm25_query"] = bm25_query
    try:
        compile(program, f"block_{self.tree.current_block_index()}", "exec")
    except Exception as e:
        block["score"] -= 2
        self.add_block(block)
        if len(self.tree.tokenizer(program)["input_ids"]) >= 768:
            raise ValueError("Length limit exceeded! Programs must be fewer than 768 tokens.")
        else:
            raise ValueError from e
    if block_type in {"orientation", "expectation", "debug"}:
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
    raw_score = asyncio.run(
        evaluate_fn([prompt[:len(prompt) - len(prefix)] + block["body"],],
                    raw=True)
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

"""
def tag_block(agent, context, block):
    render = render_block(block, tags=False)
    searcher = agent.bm25_index.searcher()
    total_blocks = index.num_docs()

    sample_size = 5
    sampled_block_ids = random.sample(range(total_blocks), sample_size)

    sampled_blocks = []
    for block_id in sampled_block_ids:
        block = searcher.doc(block_id)
        block_dict = {
            "type":block["type"],
            "q":block["q"],
            "score":block["score"],
            "render":block["render"]
            "index":block["index"],
            "timestamp":block["timestamp"],
            "tags":block["tags"]
        }

            
        sampled_blocks.append(block_dict)
        
    few_shot_prompt = ''.join([f"{block['render']}\n#tags: {block['tags']}\n"
                               for block in sampled_blocks])
    few_shot_prompt += (render + "\n#tags: ")
    guess = generate_outputs_vllm(
"""
