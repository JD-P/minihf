import time
import re
import ast
import random
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
    bm25_prompt += "# Examples\n"
    bm25_prompt += "# Retrieve blocks with 'key' in rendered body and 'solution' in tags\n"
    bm25_prompt += "#bm25_query render:key tags:solution\n"
    bm25_prompt += "# Retrieve high scoring blocks of same type\n" 
    bm25_prompt += "#bm25_query type:'{block_type}'\n"
    bm25_prompt += "# Retrieve examples where the current_task is updated\n"
    bm25_prompt += "#bm25_query type:'task-inference' render:agent.current_task render:Update\n"
    bm25_prompt += "# Retrieve observations relating to something Amanda said\n"
    bm25_prompt += "#bm25_query type:'observation' render:Amanda render:amanda render:she render:said render:remember render:forget\n"
    bm25_prompt += "# Now I'll write the query that will help me write the next block.\n"
    if self.current_block_index < 50:
        bm25_prompt += f"#bm25_query type:'{block_type}'"
    else:
        bm25_prompt += "#bm25_query "
        
    port = 5001
    # TODO: Rejection sample this?
    bm25_query = " " + generate_outputs_vllm(self.model_name,
                                             bm25_prompt,
                                             256,
                                             port=port,
                                             stop=["\n",])[0].strip()
    #type_filter_re = r"type:(\w+)"
    #_match = re.search(type_filter_re, bm25_query)
    #if _match:
    #    type_value = _match.group(1)
    #else:
    #    type_value = None

    #bm25_query = re.sub(type_filter_re, "", bm25_query).strip()

    searcher = self.bm25_index.searcher()
    try:
        query = self.bm25_index.parse_query(bm25_query, ["render", "tags"])
    except ValueError as e:
        msg = ""
        msg += "BM25 query parse failed during block generation."
        msg += f"\"{bm25_query}\" is not a valid tantivy query."
        msg += "You can resolve this by writing a valid BM25 query on your next go."
        raise ValueError(msg)
    #if type_value:
    #    type_filter = self.bm25_schema.TermFilter("type", type_value)
    #    results = searcher.search(query, limit=25, filter=type_filter).hits
    #else:
    results = searcher.search(query, limit=25).hits
    retrieved_blocks = [searcher.doc(result[1]) for result in results
                        if searcher.doc(result[1])["score"][0] >= 3]
    retrieved_blocks = sorted(retrieved_blocks,
                              key=lambda block: block["score"][0],
                              reverse=True)[:3]

    prompt = f'<s> [INST] {context}'
    if self.current_block_index > 10 and retrieved_blocks:
        prompt += f"# START RETRIEVED BLOCKS FOR BLOCK #{self.current_block_index}\n"    
        for block in retrieved_blocks:
            block_text = block["render"][0].replace(
                block["type"][0],
                "recalled-" + block["type"][0]
            )
            prompt += "\n" + block_text
        prompt += f"\n# END RETRIEVED BLOCKS FOR BLOCK #{self.current_block_index}\n"
    prompt += f"\nWrite the next {block_type} block."
    prompt += f" [/INST]#startblock type: {block_type}\n{hint}\n\n"

    # Narrow incidence of structurally wrong blocks by premising correct prefix
    if block_type in {"orientation", "expectation",
                      "task-inference", "observation_inference"}:
        prefix = '"""'
    elif block_type in {"action", "evaluation"}:
        prefix = "def "
    else:
        prefix = ""
    if block_type in {"orientation"} and self.debugging:
        with open("/app/error_stems.txt") as infile:
            error_stem = random.choice(infile.readlines())
            prefix += error_stem.strip().format(stage=self.failure_stage) + " "
    prompt += prefix
    stopstrings = ["\n#q: ", "\n# q:", "#endblock", "#startblock"]
    generate_fn = partial(generate_outputs_vllm,
                          self.model_name,
                          port=port,
                          stop=stopstrings)
    score_prompt_fns = []
    # TODO: Use the full set of questions somehow?
    score_prompt_fns.append(make_simple_score_prompt(eval_questions[0]))
    async def evaluate_fn(texts):
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
    if (branches[-1].score < 3.0):
        do_long = True
    try:
        program = branches[-1].branch_text()
        program = prefix + program.strip()
        compile(program, f"block_{self.current_block_index}", "exec")
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
             "bm25_query":bm25_query,
             "body":program,
             "q":eval_questions[0],
             "score":branches[-1].score.item()}
    self.add_block(block)
    try:
        compile(program, f"block_{self.current_block_index}", "exec")
    except Exception as e:
        raise ValueError from e
    rprint(f"Finished writing block #[cyan]{self.current_block_index}[/cyan] of type [cyan]{block_type}[/cyan]")
    print(block["body"])
    return block

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
