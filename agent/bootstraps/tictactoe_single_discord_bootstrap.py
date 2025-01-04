import requests
import json
import threading
import time
from http.server import HTTPServer
from bootstraps.tictactoe_server import TicTacToeHandler
from tools.discord import WeaveDiscordClient

# Start the server in a separate thread
server = HTTPServer(('localhost', 8000), TicTacToeHandler)
server_thread = threading.Thread(target=server.serve_forever)
server_thread.daemon = True
server_thread.start()
time.sleep(1)  # Give the server some time to start

# Start a new game against the basic AI
response = requests.post("http://localhost:8000/start", json={"ai": "basic"})
assert response.status_code == 200

#startblock type: orientation
#timestamp 1724982545.6534579
"""
WEAVER [P: EXPECTATION], I'm in a game of tic tac toe against a dumb opponent.
I want to win the game and then return to parent. The game is being played
on a HTTP server served on localhost 8000.

WEAVER [P: CLARIFICATION], How do I make a move?

WEAVER [P: EXPOSITION], You make a move using the /move endpoint and the requests
library. For example: `requests.post("http://localhost:8000/move", json={"move": 4})`
lets us take the center of the board.

WEAVER [P: CLARIFICATION], How do I get the board state?

WEAVER [P: EXPOSITION], You use the /board endpoint, which returns a JSON in this
format: {"board": ["O", " ", " ", " ", "X", " ", " ", " ", " "]} Keep in mind that
the empty spaces on the board are a space string rather than none or empty string.

WEAVER [P: RATIONAL], And I get the first move, so I can take the center?

WEAVER [P: EXPOSITION], Yes, we can take the center of the board.

WEAVER [P: CONCLUSION], Alright then I will use the following strategy:
1. Make a move to take the center of the board since that's the best opening move.
2. Continue making moves based on the current state of the board.

I will use the /board endpoint to observe the current state of the board.
Once I have won or run out of time I will return to parent.
"""
#endblock
#startblock type: action
#timestamp 1724982671.1163404
"""
I'll set up the main agent to play the rest of the game and make the first move to 
take the center of the board.
"""

def main_agent_and_move(subagent):
    def check_beaten_ai(subagent, ai_type):
        """Check if the specified AI has been beaten twice in a row."""
        response = requests.get("http://localhost:8000/history?n=10")
        if response.status_code != 200:
            return False

        games = response.json()
        ai_wins = [game for game in games if game[4] == ai_type and game[3] == 'X']
        return len(ai_wins) >= 1

    # Avoid using lambda because we need named unit test to satisfy return_to_caller
    def victory(subagent):
        return check_beaten_ai(subagent, "basic_ai")
    
    schema = {"victory":"boolean"}
    main_agent = agent.subagent("main", None, "Beat basic AI at Tic-Tac-Toe", schema, args.budget)
    main_agent.task.add_evaluation("Check beaten basic AI", victory)

    def view_current_board(subagent):
        response = requests.get("http://localhost:8000/board")
        if response.status_code == 200:
            return response.json()["board"]
        return None

    # Add the new view to the main agent instead of self since it's about to be executed
    main_agent.add_observation_view("View current game board", view_current_board)

    # Add discord communication line to bot operator
    with open("discord.json") as infile:
        credentials = json.load(infile)
        chat = WeaveDiscordClient(main_agent, credentials["key"], credentials["cid"])
        time.sleep(5)
        chat.send_message("The weave-agent is online. Taking board center...")
        # Note to self: I can access the chat object to send a message by doing
        # self.tools[f"discord-bot-{channel_id}"] where channel id is taken from
        # the top of the observation window associated with the chat.
        
    # Make the first move to take the center of the board
    move_response = requests.post("http://localhost:8000/move", json={"move": 4})
    if move_response.status_code != 200:
        return False
    return True

self.add_action("Set up main agent and make first move", main_agent_and_move)
#endblock
#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the game starts and the first move is made successfully.
The action should fail if the game does not start or the first move is not made successfully.
"""
#endblock
#startblock type: evaluation
#timestamp 1724983062.124238

def check_game_started(subagent):
    """Check if the game has started successfully."""
    response = requests.get("http://localhost:8000/board")
    if response.status_code == 200:
        board = response.json()["board"]
    else:
        return False
    return board[4] == 'X'

# Add action evaluations
self.add_evaluation(
    "Check game started",
    check_game_started
)
#endblock
