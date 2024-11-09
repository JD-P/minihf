import unittest
from block_generators import extract_function_and_add_action_or_evaluation

class TestExtractFunctionAndAddAction(unittest.TestCase):

    def test_normal_action_block(self):
        code = """
def example_function(agent):
    pass

agent.add_action('Example Action', example_function)
"""
        function_code, add_action_code = extract_function_and_add_action_or_evaluation(code, "add_action")
        expected_function_code = """
def example_function(agent):
    pass
"""
        expected_add_action_code = "agent.add_action('Example Action', example_function)"
        self.assertEqual(function_code.strip(), expected_function_code.strip())
        self.assertEqual(add_action_code.strip(), expected_add_action_code.strip())

    def test_action_block_with_unrelated_code_after(self):
        code = """
def example_function(agent):
    pass

agent.add_action('Example Action', example_function)

# Unrelated code
def unrelated_function():
    pass

print("Unrelated statement")
"""
        function_code, add_action_code = extract_function_and_add_action_or_evaluation(code, "add_action")
        expected_function_code = """
def example_function(agent):
    pass
"""
        expected_add_action_code = "agent.add_action('Example Action', example_function)"
        self.assertEqual(function_code.strip(), expected_function_code.strip())
        self.assertEqual(add_action_code.strip(), expected_add_action_code.strip())

    def test_action_block_with_unrelated_expression_between(self):
        code = """
def example_function(agent):
    pass

# Unrelated expression
print("Unrelated expression")

agent.add_action('Example Action', example_function)
"""
        function_code, add_action_code = extract_function_and_add_action_or_evaluation(code, "add_action")
        expected_function_code = """
def example_function(agent):
    pass
"""
        expected_add_action_code = "agent.add_action('Example Action', example_function)"
        self.assertEqual(function_code.strip(), expected_function_code.strip())
        self.assertEqual(add_action_code.strip(), expected_add_action_code.strip())

    def test_action_block_with_unrelated_statement_between(self):
        code = """
def example_function(agent):
    pass

# Unrelated statement
x = 42

agent.add_action('Example Action', example_function)
"""
        function_code, add_action_code = extract_function_and_add_action_or_evaluation(code, "add_action")
        expected_function_code = """
def example_function(agent):
    pass
"""
        expected_add_action_code = "agent.add_action('Example Action', example_function)"
        self.assertEqual(function_code.strip(), expected_function_code.strip())
        self.assertEqual(add_action_code.strip(), expected_add_action_code.strip())

if __name__ == '__main__':
    unittest.main()
