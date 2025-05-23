import random
from hashlib import sha256
from tools.nano import WeaveNano

#startblock type: orientation
#timestamp 1747872850.9012158
"""
Unix Environment, Nano Editor, Deliberate Practice, Feedback Loop, Verifiable Rewards,
Loop Setup, Reward Design, How To Design Reward For Nano Editor Practice, End.

I want to practice using the nano editor and reward myself using my evaluation 
blocks. What's the simplest training loop I could use? I could have a subroutine
that corrupts a text file and then write out a stream of nano editor commands that
repair it. I could do this repeatedly to teach myself how to use the editor. OK.
But I'd have to be careful to only use the nano editor to edit the file or it would
defeat the purpose of the exercise. What about a multi-layer check? I could check
both that the hash of a particular nano editor screen and a hash of the underlying
text file match reference hashes for both. Hm, OK. Should work.

Wait. How am I going to corrupt the text file? I could start simple and randomly
corrupt one of the ASCII characters into a different ASCII character. Hm. Some of
the characters in ASCII are nonprintable though, so I'd want to restrict the range
to printable 'normal' characters. What's the range of printable characters again?

> Search: What's the range of printable characters in ASCII?
>>> Decimal range 32 through 126 is printable in the 1967 edition of the ASCII
>>> standard.

Hm. Python `bytes()` objects aren't mutable so I'll have to convert to a mutable
type like a list, replace one of the characters at random and then correct it to
the expected text. Wait. If I replace one of the characters in the file then my
nano editor won't update to include it by default, so it probably makes the most
sense to close and recreate the editor between blocks. But, if I do that then it
seems probable I'll learn to set up and tear down the editor on each action block
out of habit, which would be bad. One potential solution would be to have the 
corruptor pick both a filename and a corruption to make it clearer that I am changing
which editor instance I use between action blocks for a reason. Another solution
would be to have the corruption function close and reopen the editor for me so that
it doesn't become a habit in action blocks. Kind of janky but seems like my best bet
tbh.

Okay, so: Create the text file. Wait. What am I going to put in the text file? It
should probably have multiple lines but otherwise be kind of placeholder. I think
it would help my practice if it did have semantic content so not lorem ipsum. But
this is also something I'm going to repeat many times in the context window so it
should probably be something I don't mind subconsciously influenced by fixing over
and over. How about this bit of Liber Augmen?

```
Pattern Capture
pattern-capture

Specifying a pattern in a model or predictor that captures all the novel 
behavior of that pattern. A pattern has been captured when it can't violate 
expectations anymore. 

Much of the problem with strategy and tactics is you have to optimize for an 
outcome, which entails structure and predictability. But you also have to 
avoid pattern capture, which requires being unpredictable, more complex than 
your opponent can hold in their head.

If you're too predictable, other agents can exploit you by simulating you in
their head until they find a gametree moveset(s) where you lose. This is one of 
the basic tools used by the rent seeking class to exploit the lower classes: Use 
your advantage in time and resources to totally capture the patterns of the lower
class mind and take them for everything not needed for malthusian survival.

One fundamental agent algorithm then is "Think in ways that avoid pattern 
capture". How would you have to think to use structure but avoid repeating 
yourself?
```

Okay, so: Write the text file, hash it, open it in nano, hash the tool render,
add the corruption function as a observation callback. Should work.
"""
#endblock
#startblock type: action
#timestamp 1747875806.3785787
def action_setup_main_agent_and_training(subagent):
    schema = {}
    "Create main agent for rest of run"
    main_agent = agent.subagent("main", None, "See pinned bootstrap block", schema, args.budget)

    "Lines of Liber Augmen excerpt"
    lines = ["Pattern Capture",
             "pattern-capture",
             "Specifying a pattern in a model or predictor that captures all the novel", 
             "behavior of that pattern. A pattern has been captured when it can't violate",
             "expectations anymore.", 
             "",
             "Much of the problem with strategy and tactics is you have to optimize for an",
             "outcome, which entails structure and predictability. But you also have to",
             "avoid pattern capture, which requires being unpredictable, more complex than",
             "your opponent can hold in their head.",
             "",
             "If you're too predictable, other agents can exploit you by simulating you in",
             "their head until they find a gametree moveset(s) where you lose. This is one of",
             "the basic tools used by the rent seeking class to exploit the lower classes: Use",
             "your advantage in time and resources to totally capture the patterns of the lower",
             "class mind and take them for everything not needed for malthusian survival.",
             "",
             "One fundamental agent algorithm then is \"Think in ways that avoid pattern",
             "capture\". How would you have to think to use structure but avoid repeating",
             "yourself?"
    ]
    WeaveNano(main_agent, "excerpt.txt")
    editor = main_agent.tools["nano-/app/excerpt.txt"]
    for line in lines:
        editor.send_command(line)
        editor.send_command("Enter")
    editor.send_command("C-o")
    editor.send_command("Enter")
    "Demonstrate backspace"
    editor.send_command("BSpace")
    editor.send_command("?")
    "Move cursor to end with page down and display cursor position at the end"
    "of actions so screen content always matches when we successfully fix the file"
    editor.send_command("PgUp")
    editor.send_command("C-c")
    screen_content = editor.render(main_agent)
    main_agent.update_cache("reference_screen",
                            sha256(screen_content.encode('utf-8')).hexdigest())
    with open("excerpt.txt") as infile:
        file_content = infile.read()
    main_agent.update_cache("reference_file",
                            sha256(file_content.encode('utf-8')).hexdigest())

    def corrupt(subagent):
        with open("excerpt.txt") as infile:
            if infile.read() != subagent.get_cache("reference_file"):
                return "No change made because the last change wasn't fixed."
        new_byte = 32 + random.randrange(94)
        with open("excerpt.txt", mode="rb") as infile:
            file_bytes = list(infile.read())
        index = random.randrange(len(file_bytes))
        old_byte = file_bytes[index]
        line_number = file_bytes[:index].count(10)
        line_pos = [i for i in reversed(file_bytes[:index])].index(10)
        file_bytes[index] = new_byte
        with open("excerpt.txt", mode="wb") as outfile:
            outfile.write(bytes(file_bytes))
        subagent.tools["nano-/app/excerpt.txt"].close()
        WeaveNano(subagent, "excerpt.txt")
        return (f"Index overwritten: line {line_number}, char {line_pos}\n"
                + f"Original byte: {old_byte} ({chr(old_byte)})\n"
                + f"Byte replaced with: {new_byte} ({chr(new_byte)})")

    main_agent.add_observation_view("Corrupt byte for next challenge", corrupt)
    return True

self.add_action("Set up main agent for nano training", action_setup_main_agent_and_training)
#endblock
#startblock type: expectation
#timestamp 1747877750.0129619
"""
If the action succeeds I expect to see the nano editor with the corrupted file.
I also expect to see an observation view with the `Index overwritten` and 
`Byte replaced with` strings and the associated variable values. For future 
evaluation blocks I want to sha256 hash the excerpt.txt file and nano screen 
content to make sure it matches the `reference_screen` and `reference_file` 
in my cache. Note that because of the way the observation view works I never
expect to actually see the fixed file. I'll know if I succeeded or not by whether
my evaluation block returns True in the outcomes block or not.
"""
#endblock
#q: Do I need to set up or tear down any observation callbacks? No. (97.341%)
