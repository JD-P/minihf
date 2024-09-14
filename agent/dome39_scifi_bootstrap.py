import selenium
from tools.editor import WeaveEditor

#startblock type: orientation
#timestamp 1724982545.6534579
"""
The task for the agent is to write a short story in the file scifi.txt about a Mars colonist adjusting to life in the colony after emigrating. The story should start with the following block:

---
layout: blogpost
title: The Monotony Of Dome 39
author: Morpheus Weaver
date: 2024-09-13
tags: [story]
---

Dome 39 was an oasis of life on a deserted planet.

The agent should then change the opening line to "Juanita laid uneasily in a bed of artificial flowers."

The agent should use the editor.py to perform these operations and then continue writing the rest of the story.
"""
#endblock
#startblock type: task_inference
#timestamp 1724982419.5427651
# Current Task (which will get printed by the task-reminder block)
# ID  : 1
# Title: Write the opening block of the story
# Description: Append the opening block to scifi.txt using the editor.
# Status: idle

# New Tasks:
agent.add_task(
    "Write the opening block of the story",
    "Append the opening block to scifi.txt using the editor.",
    "idle"
)

agent.add_task(
    "Change the opening line of the story",
    "Edit the opening line of the story in scifi.txt using the editor.",
    "blocked",
    blocked_on=[1,]
)

agent.add_task(
    "Write the rest of the story",
    "Continue writing the rest of the story in scifi.txt using the editor.",
    "blocked",
    blocked_on=[2,]
)

# Root Task:
agent.add_task(
    "Write a short story about a Mars colonist",
    """
    I'm writing a short story about a Mars colonist adjusting to life in the colony after emigrating.
    The story should be written in the file scifi.txt.
    """,
    "blocked",
    blocked_on=[1, 2, 3]
)

# Updates to Existing Task
agent.current_task = agent.tasks.get_task(1)
agent.current_task.going("Writing the opening block of the story...")

agent.current_task = agent.tasks.get_task(1)
#endblock
#startblock type: action
#timestamp 1724982671.1163404
"""I'll open the editor on scifi.txt and append the opening block of the story."""

def append_opening_block(agent):
    editor = WeaveEditor(agent, "scifi.txt")
    opening_block = """
---
layout: blogpost
title: The Monotony Of Dome 39
author: Morpheus Weaver
date: 2024-09-13
tags: [story]
---

Dome 39 was an oasis of life on a deserted planet.
"""
    editor.append(opening_block)
    new_opening_line = "Juanita laid uneasily in a bed of artificial flowers."
    editor.edit(9, 9, new_opening_line)
    editor.close()
    return True

agent.add_action("Append opening block", append_opening_block)
#endblock
#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the opening block is appended to scifi.txt.
The action should fail if the file is not opened or the block is not appended.
The action should succeed if the opening line is changed to "Juanita laid uneasily in a bed of artificial flowers."
The action should fail if the file is not opened or the line is not changed.
"""
#endblock
#startblock type: observation_inference
#timestamp 1725697638.8825483
"""I should open an observation monitor for the scifi.txt file to keep track of its content."""

def monitor_scifi_file(agent):
    with open("scifi.txt", "r") as file:
        return file.read()

agent.add_observation_view("Monitor scifi.txt content", monitor_scifi_file)
#endblock
#startblock type: evaluation
#timestamp 1724983062.124238

# Task Evaluations

def check_opening_block(agent):
    with open("scifi.txt", "r") as file:
        content = file.read()
        return "Dome 39 was an oasis of life on a deserted planet." in content

def check_opening_line_change(agent):
    with open("scifi.txt", "r") as file:
        content = file.read()
        return "Juanita laid uneasily in a bed of artificial flowers." in content

def evaluate_story_beginning(agent):
    question = "Does the story have a clear beginning?"
    with open("scifi.txt", "r") as file:
        story = file.read()
    score_prompt_fn = make_simple_score_prompt(question)
    score = simple_evaluate_outputs(score_prompt_fn, story)[0].item()
    return score >= 3

def evaluate_story_engagement(agent):
    question = "Is the story engaging and interesting?"
    with open("scifi.txt", "r") as file:
        story = file.read()
    score_prompt_fn = make_simple_score_prompt(question)
    score = simple_evaluate_outputs(score_prompt_fn, story)[0].item()
    return score >= 3

def evaluate_story_accuracy(agent):
    question = "Does the story accurately depict life in a Mars colony?"
    with open("scifi.txt", "r") as file:
        story = file.read()
    score_prompt_fn = make_simple_score_prompt(question)
    score = simple_evaluate_outputs(score_prompt_fn, story)[0].item()
    return score >= 3

def evaluate_story_character(agent):
    question = "Is the main character, Juanita, well-developed?"
    with open("scifi.txt", "r") as file:
        story = file.read()
    score_prompt_fn = make_simple_score_prompt(question)
    score = simple_evaluate_outputs(score_prompt_fn, story)[0].item()
    return score >= 3

def evaluate_story_conclusion(agent):
    question = "Does the story have a satisfying conclusion?"
    with open("scifi.txt", "r") as file:
        story = file.read()
    score_prompt_fn = make_simple_score_prompt(question)
    score = simple_evaluate_outputs(score_prompt_fn, story)[0].item()
    return score >= 3

task1 = agent.tasks.get_task(1)
task1.add_evaluation("Opening block is appended", check_opening_block)

task2 = agent.tasks.get_task(2)
task2.add_evaluation("Opening line is changed", check_opening_line_change)

task4 = agent.tasks.get_task(4)
task4.add_evaluation("Story has a clear beginning", evaluate_story_beginning)
task4.add_evaluation("Story is engaging and interesting", evaluate_story_engagement)
task4.add_evaluation("Story accurately depicts life in a Mars colony", evaluate_story_accuracy)
task4.add_evaluation("Main character, Juanita, is well-developed", evaluate_story_character)
task4.add_evaluation("Story has a satisfying conclusion", evaluate_story_conclusion)

# Action Evaluations

def check_scifi_file_exists(agent):
    return os.path.exists("scifi.txt")

agent.add_evaluation(
    "Check scifi.txt Exists",
    check_scifi_file_exists
)

def check_opening_block_appended(agent):
    with open("scifi.txt", "r") as file:
        content = file.read()
        return "Dome 39 was an oasis of life on a deserted planet." in content

agent.add_evaluation(
    "Check Opening Block Appended",
    check_opening_block_appended
)

def check_opening_line_changed(agent):
    with open("scifi.txt", "r") as file:
        content = file.read()
        return "Juanita laid uneasily in a bed of artificial flowers." in content

agent.add_evaluation(
    "Check Opening Line Changed",
    check_opening_line_changed
)
#endblock
