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
            if block_type == "action":
                self.assertEqual(lint_block(block_type, ""), 1.5)
            else:
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

self.add_action('Example Action', example_function)
"""

        action2 = """
def example_function(agent):
    pass

self.add_action('Example Action', example_function)
"""
        
        evaluation = """
def example_function(subagent):
    pass

self.add_evaluation('Is this an example evaluation?', example_function)
"""
        
        self.assertEqual(lint_block("action", action), 0.5)
        self.assertEqual(lint_block("action", action2), 0.5)
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

    def test_non_question_evaluation_title(self):
        evaluation = """
def example_function(subagent):
    pass

self.add_evaluation('Is this an example evaluation', example_function)
"""
        self.assertEqual(lint_block("evaluation", evaluation), 1.0)

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

    # New test cases for assertion penalties
    def test_action_no_assertions(self):
        action = """
def example_function(agent):
    pass

self.add_action('Example Action', example_function)
"""
        self.assertEqual(lint_block("action", action), 0.5)

    def test_action_assert_without_message(self):
        action = """
def example_function(agent):
    assert True

self.add_action('Example Action', example_function)
"""
        self.assertEqual(lint_block("action", action), 0.3)

    def test_action_assert_message_not_question(self):
        action = """
def example_function(agent):
    assert True, "This is a message"

self.add_action('Example Action', example_function)
"""
        self.assertEqual(lint_block("action", action), 0.2)

    def test_action_assert_in_try(self):
        action = """
def example_function(agent):
    try:
        assert True
    except:
        pass

self.add_action('Example Action', example_function)
"""
        self.assertEqual(lint_block("action", action), 0.5)

    def test_action_multiple_assert_penalties(self):
        action = """
def example_function(agent):
    try:
        assert True
        assert False, "This is not a question"
    except:
        pass

self.add_action('Example Action', example_function)
"""
        self.assertEqual(lint_block("action", action), 0.7)

    def test_evaluation_with_assertions(self):
        evaluation = """
def example_function(subagent):
    assert True

self.add_evaluation('Is this an example evaluation?', example_function)
"""
        self.assertEqual(lint_block("evaluation", evaluation), 0.0)

    def test_evaluation_with_mismatched_callback_name(self):
        evaluation = """
def example_function(subagent):
    assert True

self.add_evaluation('Is this an example evaluation?', new_function)
"""
        self.assertEqual(lint_block("evaluation", evaluation), 2.0)

    def test_real_evaluation_with_mismatched_callback_name(self):
        evaluation = """
def check_move_to_warehouse(agent):
    game_state = agent.tools['zombie_game'].last_state
    return game_state['location'] == 'WAREHOUSE'

self.add_evaluation('Did we prepare for night defense?', check_night_defense)
"""
        self.assertEqual(lint_block("evaluation", evaluation), 2.0)
        
if __name__ == '__main__':
    unittest.main()
