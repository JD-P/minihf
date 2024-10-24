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
    element_count_penalty = min(0.02 * element_count, 1)

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
    else:
        pattern_penalty = 0

    # Combine the element count penalty and the pattern penalty
    total_penalty = element_count_penalty + pattern_penalty

    return total_penalty

def compute_pattern_penalty(text):
    """Calculate the Python-like pattern penalty"""
    pattern_count = sum(len(re.findall(pattern, text)) for pattern in python_patterns)
    pattern_penalty = min(0.2 * pattern_count, 1.0)
    return pattern_penalty

def lint_block(block_type, body):
    if block_type == "orientation":
        return heuristic_penalty(body)
    else:
        return 0
