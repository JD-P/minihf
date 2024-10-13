import selenium
from tools.editor import WeaveEditor

#startblock type: orientation
#timestamp 1724982545.6534579
"""
The task for the agent is to write a horror story in the file horror.txt about four friends who walk into a haunted mansion and encounter a ghost. The story should start with the following block:

---
layout: blogpost
title: The Haunting of Mansion X
author: Morpheus Weaver
date: 2024-10-31
tags: [story, horror]
---

The agent should then change the opening line to "The four friends stood at the entrance of the mansion, feeling a chill run down their spines."

The agent should use the editor.py to perform these operations and then continue writing the rest of the story.
"""
#endblock
#startblock type: task_inference
#timestamp 1724982419.5427651
# Current Task (which will get printed by the task-reminder block)
# ID  : 1
# Title: Write the opening block of the story
# Description: Append the opening block to horror.txt using the editor.
# Status: idle

# New Tasks:
agent.add_task(
    "Write the opening block of the story",
    "Append the opening block to horror.txt using the editor.",
    "idle"
)

agent.add_task(
    "Change the opening line of the story",
    "Edit the opening line of the story in horror.txt using the editor.",
    "blocked",
    blocked_on=[1,]
)

agent.add_task(
    "Write the rest of the story",
    "Continue writing the rest of the story in horror.txt using the editor.",
    "blocked",
    blocked_on=[2,]
)

# Root Task:
agent.add_task(
    "Write a horror story about four friends encountering a ghost",
    """
    I'm writing a horror story about four friends who walk into a haunted mansion and encounter a ghost.
    The story should be written in the file horror.txt.
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
"""I'll open the editor on horror.txt and append the opening block of the story."""

def append_opening_block(agent):
    editor = WeaveEditor(agent, "horror.txt")
    opening_block = """
---
layout: blogpost
title: The Haunting of Mansion X
author: Morpheus Weaver
date: 2024-10-31
tags: [story, horror]
---

The four friends stood at the entrance of the mansion, feeling a chill run down their spines.
"""
    editor.append(opening_block)
    new_opening_line = "The four friends stood at the entrance of the mansion, feeling a chill run down their spines."
    editor.edit(9, 9, new_opening_line)
    editor.close()
    return True

agent.add_action("Append opening block", append_opening_block)
#endblock
#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the opening block is appended to horror.txt.
The action should fail if the file is not opened or the block is not appended.
The action should succeed if the opening line is changed to "The four friends stood at the entrance of the mansion, feeling a chill run down their spines."
The action should fail if the file is not opened or the line is not changed.
"""
#endblock
#startblock type: observation_inference
#timestamp 1725697638.8825483
"""I should open an observation monitor for the horror.txt file to keep track of its content."""

def monitor_horror_file(agent):
    with open("horror.txt", "r") as file:
        return file.read()

agent.add_observation_view("Monitor horror.txt content", monitor_horror_file)
#endblock
#startblock type: evaluation
#timestamp 1724983062.124238

# Task Evaluations

def check_opening_block(agent):
    with open("horror.txt", "r") as file:
        content = file.read()
        return "The four friends stood at the entrance of the mansion, feeling a chill run down their spines." in content

def check_opening_line_change(agent):
    with open("horror.txt", "r") as file:
        content = file.read()
        return "The four friends stood at the entrance of the mansion, feeling a chill run down their spines." in content

def evaluate_story_beginning(agent):
    question = "Does the story have a clear beginning?"
    with open("horror.txt", "r") as file:
        story = file.read()
    score_prompt_fn = make_simple_score_prompt(question)
    score = simple_evaluate_outputs(score_prompt_fn, story)[0].item()
    return score >= 3

def evaluate_story_engagement(agent):
    question = "Is the story engaging and interesting?"
    with open("horror.txt", "r") as file:
        story = file.read()
    score_prompt_fn = make_simple_score_prompt(question)
    score = simple_evaluate_outputs(score_prompt_fn, story)[0].item()
    return score >= 3

def evaluate_story_accuracy(agent):
    question = "Does the story accurately depict a haunted mansion and ghost encounter?"
    with open("horror.txt", "r") as file:
        story = file.read()
    score_prompt_fn = make_simple_score_prompt(question)
    score = simple_evaluate_outputs(score_prompt_fn, story)[0].item()
    return score >= 3

def evaluate_story_character(agent):
    question = "Are the four friends well-developed characters?"
    with open("horror.txt", "r") as file:
        story = file.read()
    score_prompt_fn = make_simple_score_prompt(question)
    score = simple_evaluate_outputs(score_prompt_fn, story)[0].item()
    return score >= 3

def evaluate_story_conclusion(agent):
    question = "Does the story have a satisfying conclusion?"
    with open("horror.txt", "r") as file:
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
task4.add_evaluation("Story accurately depicts a haunted mansion and ghost encounter", evaluate_story_accuracy)
task4.add_evaluation("Four friends are well-developed characters", evaluate_story_character)
task4.add_evaluation("Story has a satisfying conclusion", evaluate_story_conclusion)

# Action Evaluations

def check_horror_file_exists(agent):
    return os.path.exists("horror.txt")

agent.add_evaluation(
    "Check horror.txt Exists",
    check_horror_file_exists
)

def check_opening_block_appended(agent):
    with open("horror.txt", "r") as file:
        content = file.read()
        return "The four friends stood at the entrance of the mansion, feeling a chill run down their spines." in content

agent.add_evaluation(
    "Check Opening Block Appended",
    check_opening_block_appended
)

def check_opening_line_changed(agent):
    with open("horror.txt", "r") as file:
        content = file.read()
        return "The four friends stood at the entrance of the mansion, feeling a chill run down their spines." in content

agent.add_evaluation(
    "Check Opening Line Changed",
    check_opening_line_changed
)
#endblock
