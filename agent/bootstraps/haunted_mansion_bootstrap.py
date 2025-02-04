from functools import partial
from tools.editor import WeaveEditor

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

WEAVER [P: EMPIRICISM], Okay but I think we should demonstrate using the weave-editor
to change text to ourselves, so that we know how to do it on subsequent blocks.

WEAVER [P: AUTHOR], Makes sense. How about we use a line like 'exampel line for 
weave-editor' and then change it to the one I proposed?

WEAVER [P: EMPIRICISM], Sounds good.

WEAVER [P: CONCLUSION], I will create the 'main' subagent, add task evaluations
for writing a decent story, and then add the front matter & opening line. I will
use the weave editor to perform these operations and then continue writing the 
rest of the story.
"""
#endblock
#startblock type: action
#timestamp 1724982671.1163404
"""I'll open the editor on horror.txt and append the opening block of the story."""

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
        if score >= target_score:
            return True
        else:
            return score

    def q1(subagent):
        return evaluate_story(
            "Does the above story have a clear beginning?",
            3,
            subagent)

    def q2(subagent):
        return evaluate_story(
            "Is the above story engaging and interesting?",
            3,
            subagent)

    def q3(subagent):
        return evaluate_story(
            "Does the above story accurately depict a haunted mansion and ghost encounter?",
            3,
            subagent)

    def q4(subagent):
        return evaluate_story(
            "Are the four friends in the story above well-developed characters?",
            3,
            subagent)

    def q5(subagent):
        return evaluate_story(
            "Does the story above have a satisfying conclusion?",
            3,
            subagent)

    schema = {"q1":"boolean",
              "q2":"boolean",
              "q3":"boolean",
              "q4":"boolean",
              "q5":"boolean",
              "story":"string"}
    main_agent = agent.subagent("main", None, "Write haunted mansion story", schema, args.budget)
    main_agent.task.add_evaluation("Story has clear beginning", q1)
    main_agent.task.add_evaluation("Story is engaging and interesting", q2)
    main_agent.task.add_evaluation("Story accurately depicts haunted mansion", q3)
    main_agent.task.add_evaluation("Story develops four friends as characters", q4)
    main_agent.task.add_evaluation("Story has a satisfying conclusion", q5)

    editor = WeaveEditor(main_agent, "horror.txt")
    editor.window_size = 100
    # Demonstrate use of WeaveEditor
    editor = main_agent.tools["editor-/app/horror.txt"] # Retrieve editor instance
    opening_lines = [
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
    editor.append("\n".join(opening_lines))
    new_opening_line = "The four friends stood at the entrance of the mansion, feeling a chill run down their spines."
    editor.edit(9, 9, new_opening_line)
    return True

self.add_action("Make 'main' subagent and append opening", add_main_start_short_story)
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
#startblock type: evaluation
#timestamp 1724983062.124238

def check_horror_file_exists(agent):
    return os.path.exists("horror.txt")

self.add_evaluation(
    "Check horror.txt Exists",
    check_horror_file_exists
)

def check_opening_block_appended(agent):
    with open("horror.txt", "r") as file:
        content = file.read()
        return "The four friends stood at the entrance of the mansion, feeling a chill run down their spines." in content

self.add_evaluation(
    "Check Opening Block Appended",
    check_opening_block_appended
)

def check_opening_line_changed(agent):
    with open("horror.txt", "r") as file:
        content = file.read()
        return "The four friends stood at the entrance of the mansion, feeling a chill run down their spines." in content

self.add_evaluation(
    "Check Opening Line Changed",
    check_opening_line_changed
)
#endblock
