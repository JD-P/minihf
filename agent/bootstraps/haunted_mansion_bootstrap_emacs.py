from functools import partial
from tools.emacs import WeaveEmacs
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

WEAVER [P: EMPIRICISM], Okay but I think we should demonstrate using emacs to edit text, so that we know how to do it on subsequent blocks.

WEAVER [P: AUTHOR], Makes sense. How about we use a line like 'example line for editing' and then change it to the one I proposed?

WEAVER [P: EMPIRICISM], Sounds good.

WEAVER [P: CONCLUSION], I will create the 'main' subagent, add task evaluations for writing a decent story, and then add the front matter & opening line. I will use emacs to perform these operations and then continue writing the rest of the story.
"""
#endblock

#startblock type: action
#timestamp 1724982671.1163404
"""I'll open Emacs on horror.txt and write the opening block of the story."""

def add_main_start_short_story(subagent):
    from weave import make_score_prompt_vllm
    
    def evaluate_story(question, target_score, subagent):
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
        score = simple_evaluate_outputs(score_prompt_fn, response)[0].item()
        return score >= target_score

    def q1(subagent): return evaluate_story("Does the story have a clear beginning?", 3, subagent)
    def q2(subagent): return evaluate_story("Is the story engaging?", 3, subagent)
    def q3(subagent): return evaluate_story("Accurately depicts haunted mansion?", 3, subagent)
    def q4(subagent): return evaluate_story("Develops four friends as characters?", 3, subagent)
    def q5(subagent): return evaluate_story("Has satisfying conclusion?", 3, subagent)

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

    # Initialize Emacs editor
    emacs = WeaveEmacs(main_agent)
    
    # Create front matter and opening line
    front_matter = [
        "---",
        "layout: blogpost",
        "title: The Haunting of Mansion X",
        "author: Morpheus Weaver",
        "date: 2024-10-31",
        "tags: [story, horror]",
        "---",
        "",
        "Example opening line of the story to replace.",
    ]
    
    # Write content using Emacs commands
    # Important: Send control sequences separately with delays
    emacs.send_command('C-x')       # Send CONTROL-X
    print(emacs.pane.capture_pane())
    time.sleep(0.3)
    emacs.send_command('C-f')       # Send CONTROL-F
    time.sleep(0.5)                  # Wait for minibuffer prompt
    emacs.send_command('horror.txt') # Type filename
    time.sleep(0.2)
    emacs.send_command('\n')         # Press Enter
    time.sleep(1)                    # Wait for file to open/create

    for line in front_matter:
        emacs.send_command(f"{line}")
        time.sleep(0.2)
        emacs.send_command('\n')     # Press Enter after each line
        time.sleep(0.1)

    # Save using explicit sequence
    emacs.send_command('C-x')
    time.sleep(0.3)
    emacs.send_command('C-s')
    time.sleep(0.5)                  # Wait for save confirmation

    # Edit line 9
    new_opening = "The four friends stood at the entrance of the mansion, feeling a chill run down their spines."
    emacs.send_command('M-g')        # Meta-g
    time.sleep(0.2)
    emacs.send_command('g')          # Second g for goto-line
    time.sleep(0.2)
    emacs.send_command('9')
    time.sleep(0.2)
    emacs.send_command('\n')
    time.sleep(0.5)                  # Wait for navigation

    # Delete line
    emacs.send_command('C-a')        # Beginning of line
    time.sleep(0.2)
    emacs.send_command('C-k')        # Kill line
    time.sleep(0.2)
    emacs.send_command('C-k')        # Kill newline (second kill)
    time.sleep(0.2)

    # Insert new line
    emacs.send_command(new_opening)
    time.sleep(0.2)
    emacs.send_command('\n')         # Final newline
    time.sleep(0.2)

    # Save again
    emacs.send_command('C-x')
    time.sleep(0.3)
    emacs.send_command('C-s')
    time.sleep(0.5)                  # Wait for save
    
    return True

self.add_action("Create story with Emacs", add_main_start_short_story)
#endblock

#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if:
- Emacs successfully creates horror.txt
- Front matter is properly formatted
- Opening line contains "chill run down their spines"
- File is saved with both front matter and story content

The action should fail if:
- Emacs fails to start or open file
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
            return "Example opening line" not in lines[8] and "chill" in lines[8]
    except:
        return False

self.add_evaluation("Line Edited", check_line_replacement)
#endblock
