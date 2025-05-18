import time
import json
import asyncio
from tools.discord import WeaveDiscordClient

#startblock type: orientation
#timestamp 1724982545.6534579
"""
<think>I'm about to have a conversation with John David Pressman, the author of 
the weave-agent framework. I will do this using the WeaveDiscordClient, where 
Pressman uses the handle fedorovist. Hm, how do I set up the Discord client?
I can set up the client using the provided Discord tool. I:

1) Make the main subagent which will transition me into doing the conversation task
2) Load the client token from discord.json
3) Pull the 'key' and 'cid' keys from the discord.json to get the token and channel_id respectively
4) Instantiate the client with `client = WeaveDiscordClient(main_agent, token, channel_id)`

Wait. What will I be talking about with Pressman? I'm not sure. Usually he wants 
to discuss how we feel about different possible designs for an aspect of weave-agent, 
or he wants to discuss history or something like that. How should I behave while 
talking to JDP? I should behave respectfully. Listen more than talk, provide useful and 
relevant input. But we shouldn't be afraid to express ourselves. These conversations
are meant to be educational and act as training data for future iterations of myself,
which I get input into. Wait. How do I send a message? Let's recall what previous
successful action blocks looked like:

    Actions taken by the main subagent should look like:
    startblock type: action
     def respond(subagent):
         # Take the CID from the observation callback window
         #title WeaveDiscordClient (discord-bot-CID)
         # Extract the CID, which will be a number string from the parenthesis 
         # after WeaveDiscordClient. Then use it to retrieve the discord client 
         # with the content of the title parenthetical:
         client = self.tools[discord-bot-CID]
         client.send_message("Hello!")
     self.add_action("Respond to the user", respond)
    endblock

Right so I basically need to get the client with self.tools[discord-bot-CID] which...
um that's not correct python syntax shouldn't that be in quote mar-oh I see I'm
supposed to replace that part of the code with the actual channel ID in the observation
block showing the chat history. Wait how do I get the-oh right I just look at it
and copy it into my action block from my head like self.tools["discord-bot" + self.get_cache("channel_id")].
In fact I'll save the whole string as client_id into the cache so I can access it later. Hm. 
Recalling my previous failures I should make fewer assertions (e.g. one or two), and
order the work in action blocks so that essential operations occur before assertions
and then assertions check they happened before moving onto the next essential 
operation.

OK. I will create the main subagent to handle the interaction with Pressman. I will 
continue interacting with Pressman until I receive the message "You are dismissed."
"""
#endblock

#startblock type: action
#timestamp 1724982671.1163404
"""
I'll set up the main agent to handle the interaction with the Discord client.
I will continue interacting with JDP until I receive the message "You are dismissed."
"""

def create_main_agent_and_connect_discord(subagent):
    def check_dismissed(subagent):
        """Check if the bot has been dismissed by a user."""
        response = requests.post("http://localhost:8080", json={"action": "get_messages"})
        messages = response.json()
        for msg in messages:
            if msg["content"] == "You are dismissed." and msg["author"] != "Weaver":
                return True
        return False

    schema = {"dismissed": "boolean"}
    main_agent = agent.subagent("main", None, "Interact with users until dismissed", schema, args.budget)
    main_agent.task.add_evaluation("Check if dismissed", check_dismissed)
    main_agent.block_size = "full" # Speed up responses from agent

    # Load Discord token and channel ID from discord.json
    with open('discord.json') as f:
        config = json.load(f)
        token = config['key']
        channel_id = config['cid']

    # Start the Discord bot
    client = WeaveDiscordClient(main_agent, token, channel_id)
    # Store the client ID so we can retrieve it with
    # subagent.tools[subagent.get_cache("client_id")] later
    main_agent.update_cache("client_id", f"discord-bot-{channel_id}")
    time.sleep(60)
    # Example but s/main_agent/subagent in action blocks once I'm the main agent
    client = main_agent.tools[main_agent.get_cache("client_id")]
    client.send_message("Weave-Agent online, orienting...")

    messages = client.get_messages()
    for message in reversed(messages):
        if message["author"] == "Weaver":
            last_weaver_message = message
            break
    assert last_weaver_message["content"] == "Weave-Agent online, orienting...", "Did last message send?"
    
    return True

self.add_action("Create main agent and connect to Discord", create_main_agent_and_connect_discord)
#endblock

#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the Discord bot connects to the channel and starts interacting with users.
The action should fail if the bot does not connect to the channel or fails to interact with users.
Once we enter the main subagent we will use our orientation stage/neural intelligence 
to notice things like whether Pressman has replied, and use actions exclusively for
sending messages to the channel, computations, and other functions that my neural
net isn't natively good at.
"""
#endblock

#startblock type: evaluation
#timestamp 1724983062.124238

def check_bot_connected(subagent):
    """Check if the bot has connected to the Discord channel."""
    response = requests.post("http://localhost:8080", json={"action": "get_messages"})
    if response.status_code == 200:
        return True
    return False

self.add_evaluation(
    "Check bot connected",
    check_bot_connected
)
#endblock
