import ast
import re

# Python-like patterns to search for within the triple-quoted string literal
python_patterns = [
    r"\nimport ",
    r'print\s*\(',
    r'\bdef\s+',
    r"':\n",
    r"\n# \[",
    r"agent\.add",
    r"agent\.agent",
    r"\[INST\]",
]

def heuristic_penalty(candidate_string):
    try:
        # Parse the candidate string into a list of elements
        elements = ast.parse(candidate_string).body
    except (SyntaxError, ValueError):
        # If parsing fails, compute the Python-like pattern penalty for the whole string
        return compute_pattern_penalty(candidate_string)

    # Calculate the element count penalty
    element_count = len(elements)
    if element_count > 1:
        element_count_penalty = min(0.02 * (element_count - 1), 1)
    else:
        element_count_penalty = 0

    if element_count >= 1:
        try:
            string_content = elements[0].value.value
            if isinstance(string_content, str):
                pattern_penalty = compute_pattern_penalty(string_content)
            else:
                # If the first element is not a string, apply the full penalty
                pattern_penalty = 1.0
        except AttributeError:
            # We can infer the first element is not a string
            pattern_penalty = 1.0
    # If there are no elements in block then first element isn't string
    elif element_count == 0:
        pattern_penalty = 1.0
    else:
        pattern_penalty = 0

    # Combine the element count penalty and the pattern penalty
    total_penalty = element_count_penalty + pattern_penalty

    return total_penalty

def compute_pattern_penalty(text):
    """Calculate the Python-like pattern penalty"""
    pattern_count = sum(len(re.findall(pattern, text)) for pattern in python_patterns)
    pattern_penalty = min(0.02 * pattern_count, 1.0)
    return pattern_penalty

def compute_callback_structure_penalty(candidate_string, slot_name):
    try:
        # Parse the candidate string into a list of elements
        tree = ast.parse(candidate_string)
        elements = tree.body
    except (SyntaxError, ValueError) as e:
        # If parsing fails, return no penalty because a separate penalty
        # exists for syntax errors
        return 0.0

    # Check if the first element is a function definition
    if len(elements) >= 1 and isinstance(elements[0], ast.FunctionDef):
        # Check if the second element is an expression with agent.add_action
        if len(elements) >= 2 and isinstance(elements[1], ast.Expr):
            expr = elements[1].value
            if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == slot_name:
                # Check if the call has two arguments: a string literal and a variable name
                if len(expr.args) == 2 and isinstance(expr.args[0], ast.Str) and isinstance(expr.args[1], ast.Name):
                    pattern_penalty = 0.0  # No penalty if the structure is correct
    if 'pattern_penalty' not in locals():
        pattern_penalty = 1.0

    if len(elements) > 2:
        element_count_penalty = min(0.02 * (element_count - 2), 1)
    else:
        element_count_penalty = 0
        
    return element_count_penalty + pattern_penalty

def lint_block(block_type, body):
    if block_type == "orientation":
        return heuristic_penalty(body)
    elif block_type == "expectation":
        return heuristic_penalty(body)
    elif block_type == "action":
        return compute_callback_structure_penalty(body, "add_action")
    elif block_type == "evaluation":
        return compute_callback_structure_penalty(body, "add_evaluation")
    # TODO: Figure out penalty function for task-inference
    # TODO: Figure out penalty function for observation-inference
    else:
        return 0
