import unittest

# Assuming the heuristic functions are in a module named `heuristic_module`
from block_linters import lint_block, heuristic_penalty, compute_callback_structure_penalty

class TestHeuristicFunctions(unittest.TestCase):

    def test_add_action_in_add_evaluation_slot(self):
        block_type = "evaluation"
        body = """
def example_function(agent):
    pass

agent.add_action('Example Action', example_function)
"""
        self.assertEqual(lint_block(block_type, body), 1.0)

    def test_empty_string_penalty(self):
        block_types = ["action", "evaluation", "expectation", "orientation"]
        for block_type in block_types:
            self.assertEqual(lint_block(block_type, ""), 1.0)

    def test_patterns_in_orientation_and_expectation(self):
        block_types = ["orientation", "expectation"]
        body = """
'''
import os
print("Hello, World!")
def example_function():
    pass
agent.add_action('Example Action', example_function)
'''
"""
        expected_penalty = 0.02 * 4
        for block_type in block_types:
            self.assertEqual(lint_block(block_type, body), expected_penalty)

    def test_well_formatted_action_or_evaluation_block(self):
        action = """
def example_function(agent):
    pass

agent.add_action('Example Action', example_function)
"""

        action2 = """
def example_function(agent):
    pass

self.add_action('Example Action', example_function)
"""
        
        evaluation = """
def example_function(agent):
    pass

agent.add_evaluation('Example Evaluation', example_function)
"""
        
        self.assertEqual(lint_block("action", action), 0.0)
        self.assertEqual(lint_block("action", action2), 0.0)
        self.assertEqual(lint_block("evaluation", evaluation), 0.0)

    def test_well_formatted_orientation_or_expectation_block(self):
        block_types = ["orientation", "expectation"]
        body = """
'''
This is a well-formatted block.
'''
"""
        for block_type in block_types:
            self.assertEqual(lint_block(block_type, body), 0.0)

    def test_non_string_only_element_in_orientation_or_expectation(self):
        block_types = ["orientation", "expectation"]
        body = """
42
"""
        for block_type in block_types:
            self.assertEqual(lint_block(block_type, body), 1.0)

    def test_non_string_first_element_with_additional_elements_in_orientation_or_expectation(self):
        block_types = ["orientation", "expectation"]
        body = """
42
'String 1'
'String 2'
'String 3'
"""
        expected_penalty = 1.0 + (0.02 * 3)
        for block_type in block_types:
            self.assertEqual(lint_block(block_type, body), expected_penalty)

if __name__ == '__main__':
    unittest.main()
