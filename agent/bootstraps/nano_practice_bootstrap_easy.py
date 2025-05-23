import random
from hashlib import sha256
from tools.nano import WeaveNano
from tools.discord import WeaveDiscordClient

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
and over. Wait. I should start with something very easy to make sure I can do it
before using complex texts. Something with one line then, relatively short. How
about this tweet from John David Pressman?

```
John David Pressman (@jd_pressman) May 3

I would in fact like the LLM agent to use its available lore and background knowledge to solve problems, and considering it's a descendant of a base model prompting itself with things like "what my professor told me before leaving grad school" is reasonable strategy.
```

Yeah. Okay, so: Write the text file, hash it, open it in nano, hash the 
tool render, add the corruption function as a observation callback. Should work.
"""
#endblock
#startblock type: action
#timestamp 1747875806.3785787
def action_setup_main_agent_and_training(subagent):
    schema = {}
    "Create main agent for rest of run"
    main_agent = agent.subagent("main", None, "See pinned bootstrap block", schema, args.budget)

    # Load Discord token and channel ID from discord.json
    with open('discord.json') as f:
        config = json.load(f)
        token = config['key']
        channel_id = config['cid']

    # Start the Discord bot so JDP can help me
    client = WeaveDiscordClient(main_agent, token, channel_id)
    # Store the client ID so we can retrieve it with
    # subagent.tools[subagent.get_cache("client_id")] later
    main_agent.update_cache("client_id", f"discord-bot-{channel_id}")
    time.sleep(10)
    # Example but s/main_agent/subagent in action blocks once I'm the main agent
    client = main_agent.tools[main_agent.get_cache("client_id")]
    client.send_message("Weave-Agent online, orienting...")
    
    "Lines of John David Pressman tweet"
    lines = ["John David Pressman (@jd_pressman) May 3",
             "Enter",
             "I would in fact like the LLM agent to use its available lore and",
             "Enter"
             "background knowledge to solve problems, and considering it's a\n",
             "descendant of a base model prompting itself with things like\n",
             "\"what my professor told me before leaving grad school\" is\n",
             "reasonable strategy."
    ]
    # Leave this unchanged because it's the ground truth
    main_agent.update_cache("original_lines", lines)
    WeaveNano(main_agent, "excerpt.txt")
    editor = main_agent.tools["nano-/app/excerpt.txt"]
    editor.send_commands(lines)
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
    # Leave this unchanged because it's the ground truth
    main_agent.update_cache("reference_screen_sha256",
                            sha256(screen_content.encode('utf-8')).hexdigest())
    with open("excerpt.txt") as infile:
        file_content = infile.read()
    # Leave this unchanged because it's the ground truth
    main_agent.update_cache("reference_file_sha256",
                            sha256(file_content.encode('utf-8')).hexdigest())

    def corrupt(subagent):
        editor = subagent.tools["nano-/app/excerpt.txt"]
        attempt = editor.render(subagent)
        with open("excerpt.txt") as infile:
            file_content = infile.read()
            file_hash = sha256(file_content.encode('utf-8')).hexdigest()
            if file_hash != subagent.get_cache("reference_file_sha256"):
                return ("No change made because the last change wasn't fixed.\n"
                        + f"Attempt:\n\n{attempt}")
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

        editor.close()
        WeaveNano(subagent, "excerpt.txt")
        return ("New exercise!\n"
                + f"Index overwritten: line {line_number}, char {line_pos}\n"
                + f"Original byte: {old_byte} - {chr(old_byte)}\n"
                + f"Byte replaced with: {new_byte} - {chr(new_byte)}")

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
