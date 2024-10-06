import os
import re
import unittest
import difflib
from editor import WeaveEditor, parse_diff, parse_diff_header, process_hunk

def generate_diff(original_lines, modified_lines):
    diff = difflib.unified_diff(original_lines, modified_lines, lineterm='\n')
    return list(diff)

class TestWeaveEditor(unittest.TestCase):
    def setUp(self):
        self.agent = type('Agent', (object,), {'tools': [], 'add_observation_view': lambda self, x, y: None, 'remove_observation_view': lambda self, x: None})()
        self.editor = WeaveEditor(self.agent, 'editor_test.txt')

    def tearDown(self):
        self.editor.close()

    def test_load_save_preserves_file(self):
        first_contents = self.editor.file_content
        self.editor.save_file("test2.txt")
        self.editor.load_file("test2.txt")
        self.assertEqual(first_contents, self.editor.file_content)
        self.editor.save_file("test2.txt")
        self.editor.load_file("test2.txt")
        self.assertEqual(first_contents, self.editor.file_content)
        self.editor.save_file("test2.txt")
        with open("editor_test.txt") as infile:
            expected = infile.read()
        with open("test2.txt") as infile:
            actual = infile.read()
        self.assertEqual(expected, actual)
        os.remove("test2.txt")

class TestWeaveEditorHunkProcessing(unittest.TestCase):
    def setUp(self):
        self.agent = type('Agent', (object,), {'tools': [], 'add_observation_view': lambda self, x, y: None, 'remove_observation_view': lambda self, x: None})()
        
    def test_carriage_returns_in_hunk(self):
        hunk_lines = [
            ' This premise is largely valid.\n',
            ' \n',
            ' For example, the belief that LLMs are largely deterministic is based on the fact\n',
            '-that the same sequence of tokens yields the sHp1\r',
            "-~7f~)/AX5?.' I%NIi''4>\r",
            '-=ure. This\n',
            '+that the same sequence of tokens yields the same result at a low temperature. This\n',
            ' line of reasoning is incorrect. \n',
            ' \n',
            ' It is true that the temperature zero results do not differ on the same sequence\n'
        ]
        edits, offset = process_hunk(hunk_lines, 1)
        self.assertEqual(edits, [(3,7,hunk_lines[2][1:] + hunk_lines[6][1:] + hunk_lines[7][1:]),])
        self.assertEqual(offset, -2)

    def test_complex_hunk(self):
        hl = [
            " And the way it's different at temperature zero is the way it would be different\n",
            ' if the sequence in question was sampled at a higher temperature. \n',
            ' \n',
            "-We're running into [what seems to me a fundamental property of complex systems](https://www.dreamsarena.com/chicken/index.php?topic=4982343):\n",
            "+We're running into [what seems to me a fundamental property of complex systems](https://www.dreamsarena.com/chicken/index.php?topic=582612):\n",
            ' The apparent cause of a thing is most often the side effect or symptom of this\n',
            '-thing, not its root cause. We see the symptom and miss its essential f:unction. \n',
            '-The temper the model is being adjusted to be more favorable to those\n',
            '-solutions (which makes sense, otherwise GPT would be much worse at doing\n',
            '-classification than it is) and *only in that sense* are you training the model\n',
            "-to generate the right answer. You're not actually giving it a 'program' to\n",
            '-generate the right answer in the way you would a normal program, you make\n',
            '-random perturbations in an area and the probability mass shifts a bit.\n',
            '-\n',
            '-Some have taken this observation to be lore options that are only feasible if they get to break out of\n',
            '+thing, not its root cause. We see the symptom and miss its essential function. \n',
            '+The temperature is a metaphor for noise added to the model that is used to break\n',
            '+tie beams (hidden state equivalences) that produce probability mass. By making your\n',
            '+generation less greedy you enable more creative solutions to tasks because the\n',
            '+model gets to explore options that are only feasible if they get to break out of\n',
            ' the existing probability mass. \n',
            ' \n',
            ' The problem of course is that at high temperatures you also lose the ability to \n',
            ' maintain a coherent story as you have to navigate your way between different\n',
            ' probability mass zones. \n',
            ' \n',
            '-So really the temperature setting determinestwo things: How well the model is\n',
            '+So really the temperature setting determines two things: How well the model is\n',
            ' going to stay in a known probability mass and how creatively it can problem solve. \n',
            ' These are in tension with each other. At temperature zero the model is 100% aligned\n',
            '-with whatever the known prior is. Even a very small percentage zero over reduces \n',
            '+with whatever the known prior is. Even a very small percentage over zero reduces \n',
            " this chance of success markedly, and the more probability mass you're trying to \n",
            ' navigate through the faster your failure rate drops off with epsilon. Conversely, \n',
            ' once the temperature is high enough the probabilistic mass of the known solution\n',
        ]
        
        ol = [
            "And the way it's different at temperature zero is the way it would be different\n",
            'if the sequence in question was sampled at a higher temperature. \n',
            '\n',
            "We're running into [what seems to me a fundamental property of complex systems](https://www.dreamsarena.com/chicken/index.php?topic=582612):\n",
            'The apparent cause of a thing is most often the side effect or symptom of this\n',
            'thing, not its root cause. We see the symptom and miss its essential function. \n',
            'The temperature is a metaphor for noise added to the model that is used to break\n',
            'tie beams (hidden state equivalences) that produce probability mass. By making your\n',
            'generation less greedy you enable more creative solutions to tasks because the\n',
            'model gets to explore options that are only feasible if they get to break out of\n',
            'the existing probability mass. \n',
            '\n',
            'The problem of course is that at high temperatures you also lose the ability to \n',
            'maintain a coherent story as you have to navigate your way between different\n',
            'probability mass zones. \n',
            '\n',
            'So really the temperature setting determines two things: How well the model is\n',
            'going to stay in a known probability mass and how creatively it can problem solve. \n',
            'These are in tension with each other. At temperature zero the model is 100% aligned\n',
            'with whatever the known prior is. Even a very small percentage over zero reduces \n',
            "this chance of success markedly, and the more probability mass you're trying to \n",
            'navigate through the faster your failure rate drops off with epsilon. Conversely, \n',
            'once the temperature is high enough the probabilistic mass of the known solution\n'
        ]

        cl = [
            "And the way it's different at temperature zero is the way it would be different\n",
            'if the sequence in question was sampled at a higher temperature. \n',
            '\n',
            "We're running into [what seems to me a fundamental property of complex systems](https://www.dreamsarena.com/chicken/index.php?topic=4982343):\n",
            'The apparent cause of a thing is most often the side effect or symptom of this\n',
            'thing, not its root cause. We see the symptom and miss its essential f:unction. \n',
            'The temper the model is being adjusted to be more favorable to those\n',
            'solutions (which makes sense, otherwise GPT would be much worse at doing\n',
            'classification than it is) and *only in that sense* are you training the model\n',
            "to generate the right answer. You're not actually giving it a 'program' to\n",
            'generate the right answer in the way you would a normal program, you make\n',
            'random perturbations in an area and the probability mass shifts a bit.\n',
            '\n',
            'Some have taken this observation to be lore options that are only feasible if they get to break out of\n',
            'the existing probability mass. \n',
            '\n',
            'The problem of course is that at high temperatures you also lose the ability to \n',
            'maintain a coherent story as you have to navigate your way between different\n',
            'probability mass zones. \n',
            '\n',
            'So really the temperature setting determinestwo things: How well the model is\n',
            'going to stay in a known probability mass and how creatively it can problem solve. \n',
            'These are in tension with each other. At temperature zero the model is 100% aligned\n',
            'with whatever the known prior is. Even a very small percentage zero over reduces \n',
            "this chance of success markedly, and the more probability mass you're trying to \n",
            'navigate through the faster your failure rate drops off with epsilon. Conversely, \n',
            'once the temperature is high enough the probabilistic mass of the known solution\n',
        ]
        
        edits, offset = process_hunk(hl, 1)
        self.assertEqual(len(edits), 4)
        self.maxDiff = None
        self.assertEqual(edits[0][2], ''.join(ol[2:5]))
        self.assertEqual(edits[1][2], ''.join(ol[4:11]))
        self.assertEqual(edits[2][2], ''.join(ol[15:18]))
        self.assertEqual(edits[3][2], ''.join(ol[18:21]))
        self.editor = WeaveEditor(self.agent, 'test_file.txt')
        self.editor.file_content = cl
        self.editor.save_file('test_file.txt')
        for edit in edits:
            self.editor.edit(*edit)
        self.assertEqual(ol, self.editor.file_content)
        
class TestWeaveEditorDiff(unittest.TestCase):
    def setUp(self):
        self.agent = type('Agent', (object,), {'tools': [], 'add_observation_view': lambda self, x, y: None, 'remove_observation_view': lambda self, x: None})()
        with open("test_file.txt", "w") as outfile:
            outfile.write("Line 1\nLine 2\nLine 3\n")
        self.editor = WeaveEditor(self.agent, 'test_file.txt')

    def tearDown(self):
        self.editor.close()
        os.remove("test_file.txt")

    def test_generate_diff(self):
        original_lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
        modified_lines = ["Line 1\n", "Modified Line 2\n", "Line 3\n"]
        diff_lines = generate_diff(original_lines, modified_lines)
        expected_diff = [
            "--- \n",
            "+++ \n",
            "@@ -1,3 +1,3 @@\n",
            " Line 1\n",
            "-Line 2\n",
            "+Modified Line 2\n",
            " Line 3\n"
        ]
        self.assertEqual(diff_lines, expected_diff)

    def test_parse_diff_header(self):
        line = "@@ -1,3 +1,3 @@\n"
        result = parse_diff_header(line)
        expected_result = (1, 3, 1, 3)
        self.assertEqual(result, expected_result)

        line = "@@ -1 +1 @@\n"
        result = parse_diff_header(line)
        expected_result = (1, 1, 1, 1)
        self.assertEqual(result, expected_result)

        line = "@@ -1,3 +1 @@\n"
        result = parse_diff_header(line)
        expected_result = (1, 3, 1, 1)
        self.assertEqual(result, expected_result)

        line = "@@ -1 +1,3 @@\n"
        result = parse_diff_header(line)
        expected_result = (1, 1, 1, 3)
        self.assertEqual(result, expected_result)

    def test_parse_diff(self):
        diff_lines = [
            "--- \n",
            "+++ \n",
            "@@ -1,3 +1,3 @@\n",
            " Line 1\n",
            "-Line 2\n",
            "+Modified Line 2\n",
            " Line 3\n"
        ]
        edits = parse_diff(diff_lines)
        expected_edits = [(1, 3, 'Line 1\nModified Line 2\nLine 3\n'),]
        self.assertEqual(edits, expected_edits)

    def test_apply_edits(self):
        original_lines = self.editor.file_content
        modified_lines = ["Line 1\n", "Modified Line 2\n", "Line 3\n"]
        diff_lines = generate_diff(original_lines, modified_lines)
        self.editor.unidiff_edit(diff_lines)
        expected_content = ["Line 1\n", "Modified Line 2\n", "Line 3\n"]
        self.assertEqual(self.editor.file_content, expected_content)

    def test_generate_diff_with_addition(self):
        original_lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
        modified_lines = ["Line 1\n", "Line 2\n", "Added Line\n", "Line 3\n"]
        diff_lines = generate_diff(original_lines, modified_lines)
        expected_diff = [
            "--- \n",
            "+++ \n",
            "@@ -1,3 +1,4 @@\n",
            " Line 1\n",
            " Line 2\n",
            "+Added Line\n",
            " Line 3\n"
        ]
        self.assertEqual(diff_lines, expected_diff)

    def test_parse_diff_with_addition(self):
        diff_lines = [
            "--- \n",
            "+++ \n",
            "@@ -1,3 +1,4 @@\n",
            " Line 1\n",
            " Line 2\n",
            "+Added Line\n",
            " Line 3\n"
        ]
        edits = parse_diff(diff_lines)
        expected_edits = [(2, 3, 'Line 2\nAdded Line\nLine 3\n'),]
        self.assertEqual(edits, expected_edits)

    def test_apply_edits_with_addition(self):
        original_lines = self.editor.file_content
        modified_lines = ["Line 1\n", "Line 2\n", "Added Line\n", "Line 3\n"]
        diff_lines = generate_diff(original_lines, modified_lines)
        self.editor.unidiff_edit(diff_lines)
        expected_content = ["Line 1\n", "Line 2\n", "Added Line\n", "Line 3\n"]
        self.assertEqual(self.editor.file_content, expected_content)

    def test_apply_edits_with_multiple_hunks(self):
        original_lines = [f"Line {i+1}\n" for i in range(25)]
        modified_lines = original_lines.copy()
        modified_lines[3] = "Modified Line 4\n"
        modified_lines[12] = "Modified Line 13\n"
        modified_lines[23] = "Modified Line 24\n"
        diff_lines = list(difflib.unified_diff(modified_lines, original_lines))
        with open("test_file.txt", "w") as outfile:
            outfile.writelines(modified_lines)
            outfile.flush()
        editor = WeaveEditor(self.agent, "test_file.txt")
        editor.unidiff_edit(diff_lines)
        self.assertEqual(original_lines, editor.file_content)

    def test_apply_edits_with_multiple_hunks_subtractions(self):
        original_lines = [f"Line {i+1}\n" for i in range(25)]
        modified_lines = original_lines.copy()
        modified_lines[3] = "Modified Line 4\n"
        modified_lines[12] = "Modified Line 13\n"
        modified_lines[23] = "Modified Line 24\n"
        del(modified_lines[6])
        del(modified_lines[7])
        del(modified_lines[20])
        diff_lines = list(difflib.unified_diff(modified_lines, original_lines))
        with open("test_file.txt", "w") as outfile:
            outfile.writelines(modified_lines)
            outfile.flush()
        editor = WeaveEditor(self.agent, "test_file.txt")
        editor.unidiff_edit(diff_lines)
        self.assertEqual(original_lines, editor.file_content)

    def test_apply_edits_with_multiple_hunks_more_subtracts_than_mods(self):
        original_lines = [f"Line {i+1}\n" for i in range(25)]
        modified_lines = original_lines.copy()
        modified_lines[3] = "Modified Line 4\n"
        modified_lines[23] = "Modified Line 24\n"
        del(modified_lines[12])
        del(modified_lines[6])
        del(modified_lines[7])
        del(modified_lines[20])
        diff_lines = list(difflib.unified_diff(modified_lines, original_lines))
        with open("test_file.txt", "w") as outfile:
            outfile.writelines(modified_lines)
            outfile.flush()
        editor = WeaveEditor(self.agent, "test_file.txt")
        editor.unidiff_edit(diff_lines)
        self.assertEqual(original_lines, editor.file_content)

    def test_apply_edits_with_multiple_hunks_subtract_start(self):
        original_lines = [f"Line {i+1}\n" for i in range(25)]
        modified_lines = original_lines.copy()
        modified_lines[3] = "Modified Line 4\n"
        modified_lines[23] = "Modified Line 24\n"
        del(modified_lines[0])
        del(modified_lines[6])
        del(modified_lines[7])
        del(modified_lines[20])
        diff_lines = list(difflib.unified_diff(modified_lines, original_lines))
        with open("test_file.txt", "w") as outfile:
            outfile.writelines(modified_lines)
            outfile.flush()
        editor = WeaveEditor(self.agent, "test_file.txt")
        editor.unidiff_edit(diff_lines)
        self.assertEqual(original_lines, editor.file_content)

    def test_apply_edits_with_multiple_hunks_more_subtracts_than_mods_adds(self):
        original_lines = [f"Line {i+1}\n" for i in range(25)]
        modified_lines = original_lines.copy()
        modified_lines[3] = "Modified Line 4\n"
        modified_lines[23] = "Modified Line 24\n"
        modified_lines.insert(9, "Added Line 10\n")
        del(modified_lines[12])
        del(modified_lines[6])
        modified_lines.insert(6, "Added Line 7\n")
        del(modified_lines[7])
        del(modified_lines[20])
        diff_lines = list(difflib.unified_diff(modified_lines, original_lines))
        with open("test_file.txt", "w") as outfile:
            outfile.writelines(modified_lines)
            outfile.flush()
        editor = WeaveEditor(self.agent, "test_file.txt")
        editor.unidiff_edit(diff_lines)
        self.assertEqual(original_lines, editor.file_content)

    def test_apply_edits_unlabeled_fuzz1(self):
        diff_lines = [
            '--- \n',
            '+++ \n',
            '@@ -16,9 +16,7 @@\n',
            ' if the sequence in question was sampled at a higher temperature. \n',
            ' \n',
            " We're running into [what seems to me a fundamental property of complex systems](https://www.dreamsarena.com/chicken/index.php?topic=582612):\n",
            '-The apparent cause of a thing is most often thhhhsYV"Cp!}\x0b',
            '-"=OEtm!\n',
            '-!^V+b7\t-this\n',
            '+The apparent cause of a thing is most often the side effect or symptom of this\n',
            ' thing, not its root cause. We see the symptom and miss its essential function. \n',
            ' The temperature is a metaphor for noise added to the model that is used to break\n',
            ' tie beams (hidden state equivalences) that produce probability mass. By making your\n',
            '@@ -73,18 +71,14 @@\n',
            ' to be like a classical program which deterministically computes that answer when \n',
            ' given that input. However, the real thing that is happening is that the\n',
            ' probability mass of the model is being adjusted to be more favorable to those\n',
            '-solutions (which makes sense, otherwise GPT would be much worse rious social \n',
            '-goals. When you come home to your own village no one has to ask ‘are you one of \n',
            '-us?’. But in a world where society is a self referencing hologram that doesn’t\n',
            '-work, it’s only as effective as the clarity of your self image and your understanding\n',
            '-of the system. \n',
            '-\n',
            "-Someone whose self image is fuzzy, who does not have sharp edges of their ownYou're not actually giving it a 'program' to\n", '+solutions (which makes sense, otherwise GPT would be much worse at doing\n',
            '+classification than it is) and *only in that sense* are you training the model\n',
            "+to generate the right answer. You're not actually giving it a 'program' to\n",
            ' generate the right answer in the way you would a normal program, you make\n',
            ' random perturbations in an area and the probability mass shifts a bit.\n',
            ' \n',
            ' Some have taken this observation to be a point against using model outputs as a\n',
            '-source of dataset generatiOn, but I do not think this is true.\n',
            '+source of dataset generation, but I do not think this is true.\n',
            ' \n',
            ' Why? Because a temperature 1 sample is a point that is maximally distant from\n',
            " probability mass. You aren't sampling from the probability mass, so it's hard to\n",
            '@@ -95,5 +89,5 @@\n',
            ' was an effectively random point at one temperature when we try and shift the\n',
            ' model to produce it deterministically.\n',
            ' \n',
            '-Put anot is for a model trained to\n',
            '+Put another way, I think we underestimate how hard it is for a model trained to\n',
            ' sample from probability mass to produce a result from the opposite side of the coin.'
        ]
        editor = WeaveEditor(self.agent, 'fuzz1.txt')
        editor.save_file('test_file.txt')
        edits = parse_diff(diff_lines)
        for start_line, end_line, new_text in edits:
            editor.edit(start_line, end_line, new_text)
        with open("editor_test.txt") as infile:
            editor_test = infile.readlines()
        self.maxDiff = None
        self.assertEqual(editor_test, editor.file_content)
        
        
if __name__ == '__main__':
    unittest.main()
