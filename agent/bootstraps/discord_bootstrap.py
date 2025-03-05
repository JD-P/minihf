import time
import json
import asyncio
from tools.discord import WeaveDiscordClient

#startblock type: orientation
#timestamp 1724982545.6534579
"""
WEAVER [P: EXPECTATION], I need to create a Discord bot that will interact with users in a specified channel.
The bot should continue interacting until a user sends the message "You are dismissed."

WEAVER [P: CLARIFICATION], How do I set up the Discord bot?

WEAVER [P: EXPOSITION], You can set up the Discord bot using the provided Discord tool.
This involves creating a Discord application, inviting the bot to the server, and configuring the bot with the necessary permissions.

WEAVER [P: RATIONAL], The bot should be able to send and receive messages in the specified channel.
It should also be able to react to messages and reply to specific messages if needed.

WEAVER [P: CONCLUSION], I will create the main subagent to handle the interaction with the Discord bot.
The bot will continue interacting with users until it receives the message "You are dismissed."
"""
#endblock

#startblock type: action
#timestamp 1724982671.1163404
"""
I'll set up the main agent to handle the interaction with the Discord bot.
The bot will continue interacting with users until it receives the message "You are dismissed."
"""

def create_main_agent_and_connect_discord(subagent):
    def check_dismissed(subagent):
        """Check if the bot has been dismissed by a user."""
        response = requests.post("http://localhost:8080", json={"action": "get_messages"})
        messages = response.json()
        for msg in messages:
            if msg["content"] == "You are dismissed." and msg["author"] != "weave-agent":
                return True
        return False

    schema = {"dismissed": "boolean"}
    main_agent = agent.subagent("main", None, "Interact with users until dismissed", schema, args.budget)
    main_agent.task.add_evaluation("Check if dismissed", check_dismissed)

    # Load Discord token and channel ID from discord.json
    with open('discord.json') as f:
        config = json.load(f)
        token = config['key']
        channel_id = config['cid']

    # Start the Discord bot
    client = WeaveDiscordClient(main_agent, token, channel_id)
    time.sleep(5)
    client.send_message("Hi!")

    # Actions taken by the main subagent should look like:
    #startblock type: action
    # def respond(subagent):
    #     # Take the CID from the observation callback window
    #     #title WeaveDiscordClient (discord-bot-CID)
    #     # Extract the CID, which will be a number string from the above title
    #     # Then use it to retrieve the discord client like below:
    #     client = self.tools[discord-bot-CID]
    #     client.send_message("Hello!")
    # self.add_action("Respond to the user", respond)
    #endblock
    # You don't have to re-initialize the Discord client or anything like that.

    return True

self.add_action("Create main agent and connect to Discord", create_main_agent_and_connect_discord)
#endblock

#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the Discord bot connects to the channel and starts interacting with users.
The action should fail if the bot does not connect to the channel or fails to interact with users.
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
