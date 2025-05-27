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
blocks. I tried a simple training loop of randomly mutating one character but
then realized there's an optimal program that solves that, which makes it kind
of pointless once you find that program. Instead it makes more sense to generate
exercises using a few shot prompt. I've put my prompt in `/app/bootstraps/nano_exercises/prompt.txt`.
I should create an observation view which sends the prompt to my underlying
generative model and then displays the next exercise to do. I create a text
file using the nano editor which satisfies that exercise and then copy the
verifier during my evaluation stage of the same tick. Wait. What happens on
the next tick? Should I only move to the next exercise after verifying that I've
done the current one? How would I detect that the current one is done from within
the weave-agent framework? Hm. I think it would probably be simpler to just move
on since the exercises are generated and by doing a different one each tick I stop
myself from getting stuck in loops or wasting time on things I don't know how to
do.

I think I should also set up the Discord client so I can talk to JDP during the
exercises, that way he can give me tips if I need help or get stuck.
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

    WeaveNano(main_agent, "demo.txt")
    editor = main_agent.tools["nano-/app/demo.txt"]
    # Add demo text
    editor.send_commands(["Hello!", "Enter", "World!"])
    # Demonstrate backspace by changing exclamation to question
    editor.send_command("BSpace")
    editor.send_command("?")
    # Save file
    editor.send_command("C-o")
    editor.send_command("Enter")

    main_agent.update_cache("exercise_num", 21)
    def next_exercise(subagent):
        with open("/app/bootstraps/nano_exercises/prompt.txt") as infile:
            prompt = infile.read().format(iteration=subagent.get_cache("exercise_num"))
        stopstrings = ["\n\n", "# Exercise", "#Exercise"]
        candidates = generate_outputs_vllm(subagent.model_name, prompt,
                                           768, port=5001, n=8, stop=stopstrings)
        # Take first candidate with valid syntax
        for candidate in candidates:
            try:
                compile(candidate, filename="candidate.py", mode="exec")
                break
            except:
                continue
        subagent.update_cache("exercise_num",
                              subagent.get_cache("exercise_num") + 1)
        return ("Write a motor action which uses the Nano editor to satisfy the conditions:\n\n"
                "# Exercise " + str(subagent.get_cache("exercise_num"))
                + candidate)

    main_agent.add_observation_view("Generate and display next nano exercise", next_exercise)
    return True

self.add_action("Set up main agent for nano training", action_setup_main_agent_and_training)
#endblock
#startblock type: expectation
#timestamp 1747877750.0129619
"""
If the action succeeds I expect to see a new exercise in the observation window.
I also expect to see a demo editor still open with the Hello! World? text.
At evaluation time on the next tick I should use the unit test shown by the
observation window to grade the motor action I took to try and satisfy its 
conditions.
"""
#endblock
#q: Do I need to set up or tear down any observation callbacks? No. (97.341%)
