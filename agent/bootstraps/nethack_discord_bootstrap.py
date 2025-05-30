import libtmux
import time
from tools.nethack import WeaveNethack
from tools.discord import WeaveDiscordClient

#startblock type: orientation
#timestamp 1724982545.6534579
"""
WEAVER [P: EXPECTATION], I need to create a weave-agent that can play and beat the game Nethack.
The game will be controlled using the WeaveNethack tool, which uses libtmux to interact with the game.

WEAVER [P: CLARIFICATION], How do I start the game and make moves?

WEAVER [P: EXPOSITION], You start the game by initializing the WeaveNethack tool and sending commands
to the tmux pane using the send_command method. For example: `nethack.send_command('h')` moves the character left.

WEAVER [P: RATIONAL], The goal is to beat the game, which is indicated by the following message:
You offer the Amulet of Yendor to [your god]...
An invisible choir sings, and you are bathed in radiance...
The voice of [your god] [booms out/booms/thunders/rings out]: "Mortal, thou hast done well!"
"In return to thy service, I grant thee the gift of Immortality!"
You ascend to the status of Demigod[dess]...

WEAVER [P: CONCLUSION], Alright then I will use the following strategy:
1. Initialize the WeaveNethack tool to start the game.
2. Make a few initial moves to demonstrate how to use the game tool.
3. Continue making moves based on the current state of the game.

I will use the render method to observe the current state of the game.
Once I have beaten the game, I will return to the parent.
"""
#endblock

#startblock type: action
#timestamp 1724982671.1163404
"""
I'll set up the main agent to play the rest of the game and make a few initial moves.
"""

def main_agent_and_initial_moves(subagent):
    def check_game_beaten(subagent):
        """Check if the game has been beaten."""
        pane_content = subagent.tools["nethack"].pane.capture_pane()
        return "You ascend to the status of Demigod" in pane_content

    schema = {"check_game_beaten": "boolean"}
    main_agent = agent.subagent("main", None, "Beat Nethack", schema, args.budget)
    main_agent.task.add_evaluation("Check if game beaten", check_game_beaten)

    nethack = WeaveNethack(main_agent)
    # To recall the tool during actions do this:
    # nethack = self.tools["nethack"]

    # Don't need to add observation view because tool already includes it
    
    # Make a few initial moves
    time.sleep(2)  # Wait for Nethack to start
    nethack.send_keys('Weaver\n')  # Input my name
    time.sleep(0.5)
    nethack.send_keys('y') # Accept random character creation
    # Leave rest blank so I can see what was chosen and whether it's acceptable
    # I should keep in mind I can get past the --more-- screens
    # with spacebar: `send_keys(' ')` or enter: `send_keys('\n')`
    # Tip: send_keys() is properly implemented, if you press j and can't go down
    # it's probably because you are trying to walk through a wall.
    # I AM SERIOUS IF YOU TRY TO GO DOWN AND CAN'T YOU ARE RUNNING INTO A WALL

    # Add discord communication line to bot operator
    with open("discord.json") as infile:
        credentials = json.load(infile)
        chat = WeaveDiscordClient(main_agent, credentials["key"], credentials["cid"])
        time.sleep(5)
        chat.send_message("The weave-agent is online. Starting nethack...")
        # Note to self: I can access the chat object to send a message by doing
        # self.tools[f"discord-bot-{channel_id}"] where channel id is taken from
        # the top of the observation window associated with the chat.
    
    return True

self.add_action("Set up main agent and make initial moves", main_agent_and_initial_moves)
#endblock

#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the game starts and the initial moves are made successfully.
The action should fail if the game does not start or the initial moves are not made successfully.
"""
#endblock

#startblock type: observation-inference
#timestamp 1737284368.573438

# No observation callbacks need to be added or removed in this run because I'm
# playing nethack, which has a dedicated tmux pane/tool I can send commands to.
# It's not possible to write observation callbacks that inspect the game state
# so no more observation callbacks than the existing one are necessary. I should
# not remove the existing observation callback because it is associated with the
# nethack tool and will wreck my game. The discord client is there to help you
# when you get stuck, it will provide expert human advice.
#
# Any time I'm prompted to add or remove an observation window I should just
# output a comment like this that one isn't necessary, and keep it short too for
# the sake of not having to generate a bunch of tokens.

#endblock
