from tools.nano import WeaveNano

#startblock type: orientation
#timestamp 1747793894.8135977
"""
Unix Environment, Python Program, Agent, Book Report,
Book Report About Sir Rowland Hill, Postal Reform,
Social Conflict, How Rowland Hill Reform Amidst Social Conflict,
Why Rowland Hill, Book Report With Text Editor, Sources Gathered,
Public Domain Books, Read The Books And Write A Report About Them,
Need To Download The Books, Books At MiniHF Website, Text File,
hill.txt, End.

Right now it's in my head rather than the context window but in the last phase
we found sources that are 1) public domain and 2) tell the story of Sir Rowland 
Hill's postal reform. Now I need to explore these sources to figure out the
answer to the research question "How did Sir Rowland Hill turn his ingenious
idea for a cheap postal system into the government granting him control of the
postal system.", my answer must be in the form of a text file, hill.txt and it
has to cite specific passages from the available resources to support my answer.

The five public domain sources I will be reviewing to try and determine the answer
are:

1) https://minihf.com/assets/Sir_Rowland_Hill_K._C._B._a_biographical_and_historical_sketch.txt
2) https://minihf.com/assets/post_office_reform_its_importance_and_practicability.txt
3) https://minihf.com/assets/pg46498.txt
4) https://minihf.com/assets/pg55467.txt
5) https://minihf.com/assets/pg59390.txt

I will download these with wget and then open them with the nano text editor to 
read.
"""
#endblock
#startblock type: action
#timestamp 1747795694.001662
def action_download_and_open_resources(subagent):
    """Set up main agent while downloading and opening the resources."""
    # Have main agent return contents of hill.txt as a string when it finishes
    schema = {"answer":"string"}
    # Create main agent for rest of run
    main_agent = agent.subagent("main", None, "See pinned bootstrap block", schema, args.budget)

    # Write out file names to prepare download and open for loop
    filenames = ["Sir_Rowland_Hill_K._C._B._a_biographical_and_historical_sketch.txt",
                 "post_office_reform_its_importance_and_practicability.txt",
                 "pg46498.txt",
                 "pg55467.txt",
                 "pg59390.txt"]
    # Store filenames for easy recall in future actions
    main_agent.update_cache("filenames", filenames)

    # Download and open files in nano editor
    for filename in filenames:
        os.system(f"wget https://minihf.com/assets/{filename}")
        WeaveNano(main_agent, filename)

    # Demonstrate retrieving and sending commands to editors
    editors = [main_agent.tools[f"nano-/app/{filename}"] for filename in filenames]
    # Send three page down keys to each instance of the nano editor
    # this way we get past the Gutenberg intros etc
    # Wait how do I use tmux send-keys again? Lets recall the docs:
    # """
    # When specifying keys, most represent themselves (for example ‘A’ to ‘Z’).
    # Ctrl keys may be prefixed with ‘C-’ or ‘^’, Shift keys with ‘S-’ and
    # Alt (meta) with ‘M-’. In addition, the following special key names are
    # accepted: Up, Down, Left, Right, BSpace, BTab, DC (Delete), End, Enter,
    # Escape, F1 to F12, Home, IC (Insert), NPage/PageDown/PgDn, PPage/PageUp/PgUp,
    # Space, and Tab.
    # """
    commands = ["PgDn", "PgDn", "PgDn"]
    for editor in editors:
        for command in commands:
            editor.send_command(command)
        
    # Open scratch.txt scratchpad and hill.txt answer file
    WeaveNano(main_agent, "scratch.txt")
    WeaveNano(main_agent, "hill.txt")

    return True

self.add_action("Set up main agent and scroll down in editors", action_download_and_open_resources)
#endblock
#startblock type: expectation
#timestamp 1747797207.3600123
"""
I expect to see five instances of the weave nano wrapper with text from books in
them. I also expect to see two more editor instances, one for a scratchpad and
one for a final answer file. My evaluation could check that each of the seven
expected editors is a valid keys in 'main_agent.tools'. I could also check for the
existence of the five downloaded files in my local working directory with 'os.path.exists()'.
"""
#endblock
#q: Do I need to set up or tear down any observation callbacks? No. (95.8%)
