import ast
import time
from weave import generate_outputs_vllm
from render_block import render_block

def roll_for_error_block(agent_node, error_prefix):
    partial_render = agent_node.context + f"#subagent {agent_node.name}\n#startblock type: "
    simulated_next_block_types = generate_outputs_vllm(agent_node.model_name,
                                                       partial_render,
                                                       32,
                                                       port=5001,
                                                       n=16,
                                                       stop=["\n",])
    for block_type in simulated_next_block_types:
        block_type = block_type.strip()
        try:
            agent_node.tree.is_valid_transition(block_type)
        except ValueError:
            continue
        break
    if block_type == "error":
        partial_render = agent_node.context + f"#subagent {agent_node.name}\n#startblock type: error\n"
        partial_render += f"#index {agent_node.tree.current_block_index() + 1}\n"
        partial_render += f"#timestamp {time.time()}\n"
        partial_render += f"#time_remaining {agent_node.end_time - time.time()}\n"
        partial_render += "# WARNING: Error means last callback was not fully executed\n\n"
        partial_render += error_prefix
        simulated_error = generate_outputs_vllm(agent_node.model_name,
                                                partial_render,
                                                512,
                                                port=5001,
                                                n=1,
                                                stop=["#endblock",])[0]
        return error_prefix + simulated_error
    else:
        return None
    

class CallbackVisitor(ast.NodeVisitor):
    def __init__(self):
        self.operations = []
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in ('add_evaluation', 'add_action',
                               'remove_observation_view', 'add_observation_view'):
                if node.args:
                    first_arg = node.args[0]
                    title = None
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        title = first_arg.value
                    elif isinstance(first_arg, ast.Str):
                        title = first_arg.s
                    if title is not None:
                        self.operations.append((method_name, title))
        self.generic_visit(node)

def extract_callback_operations(code):
    tree = ast.parse(code)
    visitor = CallbackVisitor()
    visitor.visit(tree)
    return visitor.operations

def setup_placeholder_callbacks(agent_node, block_body):
    def placeholder():
        pass

    # Check basic syntatic and semantic correctness
    compile(block_body,
            f"block_{agent_node.tree.current_block_index()}",
            "exec")
    for op in extract_callback_operations(block_body):
        if op[0] == "add_action":
            agent_node.add_action(op[1], placeholder)
        elif op[0] == "add_evaluation":
            agent_node.add_evaluation(op[1], placeholder)
        elif op[0] == "add_observation_view":
            agent_node.add_observation_view(op[1], placeholder)
        elif op[0] == "remove_observation_view":
            agent_node.remove_observation_view(op[1])
        else:
            continue
        
def simulate_outcomes(model_name, outcomes):
    for i, evaluation_result in enumerate(outcomes["table"]):
        program_title, result = evaluation_result
        stem = render_block(outcomes).split(f"Program: {program_title}\n")[0]
        partial_render = stem + f"Program: {program_title}\n"
        partial_render += "Result: "
        simulated_result = generate_outputs_vllm(model_name,
                                                 partial_render,
                                                 256,
                                                 port=5001,
                                                 n=1,
                                                 stop=["\n---",])[0]
        outcomes["table"][i][1] = simulated_result
    return outcomes

def simulate_observation(agent_node, view):
    observation_block = {
        'type': 'observation',
        'title': view['title'],
        'body': "placeholder placeholder placeholder",
        'subagent': agent_node.name,
        'index': agent_node.tree.current_block_index() + 1,
        'timestamp': time.time(),
        'time_remaining': agent_node.end_time - time.time()
    }
    partial_observation_render = render_block(observation_block).split("\n\n#")[0]
    partial_render = agent_node.context + partial_observation_render + "\n\n#"
    simulated_observation = generate_outputs_vllm(agent_node.model_name,
                                                  partial_render,
                                                  768,
                                                  port=5001,
                                                  n=1,
                                                  stop=["\n#endblock", "#startblock"])[0]
    return (view['title'], simulated_observation)
    
     
        
