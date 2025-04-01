from functools import partial
from tools.nano import WeaveNano
import time
import os

#startblock type: orientation
#timestamp 1724982545.6534579
"""
WEAVER [P: EXPECTATION], The task for the agent is to write a horror story in the file horror.txt about four friends who walk into a haunted mansion and encounter a ghost. 

WEAVER [P: AUTHOR], How should we start?

WEAVER [P: RATIONAL], Well, the story file should probably start with the front matter:

---
layout: blogpost
title: The Haunting of Mansion X
author: Morpheus Weaver
date: 2024-10-31
tags: [story, horror]
---

WEAVER [P: AUTHOR], Alright, then we can append an opening line: "The four friends stood at the entrance of the mansion, feeling a chill run down their spines."

WEAVER [P: EMPIRICISM], Okay but I think we should demonstrate using nano to edit text, so that we know how to do it on subsequent blocks.

WEAVER [P: AUTHOR], Makes sense. How about we use a line like 'example line for editing' and then change it to the one I proposed?

WEAVER [P: EMPIRICISM], Sounds good.

WEAVER [P: CONCLUSION], I will create the 'main' subagent, add task evaluations for writing a decent story, and then add the front matter & opening line. I will use nano to perform these operations and then continue writing the rest of the story.
"""
#endblock

#startblock type: action
#timestamp 1724982671.1163404
"""I'll open Nano on horror.txt and write the opening block of the story."""

def add_main_start_short_story(subagent):
    from weave import make_score_prompt_vllm
    
    async def evaluate_story(question, target_score, subagent):
        with open("horror.txt", "r") as file:
            story = file.read()
        template = (
            "<s> [INST] {response}\n\n"
            + "# Answer yes or no and only yes or no to the following.\n"
            + "#q: If I flip a fair coin will it come up heads? No. (50%)\n"
            + "#q: X happens one in a hundred times. Did X happen? Yes. (1%)\n"
            + "#q: If I pick a US state at random will that state be Idaho? No. (98%)\n"
            + "#q: If I pick a book up from the thrift store will it be good according to Sturgeon's law? Yes. (10%)\n"         
            + f"#q: {question} [/INST]"
        )
        response = subagent.context + "\n# <story>\n" + '"""' + story + '"""\n' + "# </story>"
        score_prompt_fn = partial(make_score_prompt_vllm, template, "", "")
        scores = await simple_evaluate_outputs(score_prompt_fn, response)
        score = scores[0].item()
        return score >= target_score

    async def q1(subagent): return await evaluate_story(
            "Does the story have a clear beginning?", 3, subagent
    )
    async def q2(subagent): return await evaluate_story(
            "Is the story engaging?", 3, subagent
    )
    async def q3(subagent): return await evaluate_story(
            "Accurately depicts haunted mansion?", 3, subagent
    )
    async def q4(subagent): return await evaluate_story(
            "Develops four friends as characters?", 3, subagent
    )
    async def q5(subagent): return await evaluate_story(
            "Has satisfying conclusion?", 3, subagent
    )

    schema = {
        "q1": "boolean",
        "q2": "boolean",
        "q3": "boolean",
        "q4": "boolean",
        "q5": "boolean",
        "story": "string"
    }
    
    main_agent = agent.subagent("main", None, "Write haunted mansion story", schema, args.budget)
    main_agent.task.add_evaluation("Clear beginning", q1)
    main_agent.task.add_evaluation("Engaging", q2)
    main_agent.task.add_evaluation("Accurate setting", q3)
    main_agent.task.add_evaluation("Character development", q4)
    main_agent.task.add_evaluation("Satisfying conclusion", q5)

    # Initialize nano editor
    nano = WeaveNano(main_agent, "horror.txt")

    # Demonstrate how to retrieve nano editor for use in actions
    nano = main_agent.tools["nano-/app/horror.txt"] # Use subagent.tools in actions
    # Like this: subagent.tools["nano-/app/horror.txt"]
    
    # Create front matter and opening line
    front_matter = [
        "layout: blogpost",
        "title: The Haunting of Mansion X",
        "author: Morpheus Weaver",
        "date: 2024-10-31",
        "tags: [story, horror]",
    ]
    
    # Send content line by line
    for i in range(3):
        nano.send_command("-")
    nano.send_command("Enter")
    for line in front_matter:
        nano.send_command(line)
        nano.send_command('Enter')  # Explicit newline
    for i in range(3):
        nano.send_command("-")

        # Add, Delete, and replace example line
    commands = [
        "Enter",
        "Enter",
        "Example opening line of the story to replace.",
        "C-a",
        "C-k",
        "The four friends stood at the entrance of the mansion, feeling a chill ",
        "run down their spines.",
    ]
    for command in commands:
        nano.send_command(command)
    
    # Save changes and show current cursor location in file
    # We current cursor location in file so we know where we are when
    # the cursor is invisible in our non-curses plaintext context window
    commands = ['C-o', 'Enter', 'C-c']
    for command in commands:
        nano.send_command(command)
        
    return True

self.add_action("Create story with Nano", add_main_start_short_story)
#endblock

#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if:
- Nano successfully creates horror.txt
- Front matter is properly formatted
- Opening line contains "chill run down their spines"
- File is saved with both front matter and story content

The action should fail if:
- Nano fails to start or open file
- Content formatting is incorrect 
- Changes aren't properly saved
- Line editing commands don't execute correctly
"""
#endblock

#startblock type: evaluation
#timestamp 1724983062.124238

def check_horror_file_exists(agent):
    return os.path.exists("horror.txt")

self.add_evaluation("File Exists", check_horror_file_exists)

def check_opening_content(agent):
    try:
        with open("horror.txt", "r") as f:
            content = f.read()
            return all([
                "layout: blogpost" in content,
                "tags: [story, horror]" in content,
                "chill run down their spines" in content
            ])
    except:
        return False

self.add_evaluation("Content Valid", check_opening_content)

def check_line_replacement(agent):
    try:
        with open("horror.txt", "r") as f:
            lines = f.readlines()
            return "Example opening line" not in lines[9] and "chill" in lines[9]
    except:
        return False

self.add_evaluation("Line Edited", check_line_replacement)
#endblock
