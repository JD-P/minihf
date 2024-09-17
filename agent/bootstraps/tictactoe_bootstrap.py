import requests
import json
import threading
import time
from http.server import HTTPServer
from bootstraps.tictactoe_server import TicTacToeHandler

# Start the server in a separate thread
server = HTTPServer(('localhost', 8000), TicTacToeHandler)
server_thread = threading.Thread(target=server.serve_forever)
server_thread.daemon = True
server_thread.start()
time.sleep(1)  # Give the server some time to start

#startblock type: orientation
#timestamp 1724982545.6534579
"""
I need to beat the tic-tac-toe server's AI opponents in order of increasing difficulty.
The opponents are: basic, random, defensive, offensive, side_preferring.
I need to beat each opponent twice in a row.

I will use the following strategy:
1. Start a game against the basic opponent.
2. Make a move to take the center of the board since that's the best opening move.
3. Continue making moves based on the current state of the board.
4. Check the game history to see if I have beaten the opponent twice in a row.
5. Move on to the next opponent if I have beaten the current opponent twice in a row.

I will use the /board endpoint to observe the current state of the board.
I will use the /history endpoint to check if I have beaten the opponent twice in a row.
"""
#endblock
#startblock type: task_inference
#timestamp 1724982419.5427651
"""
This is the bootstrap block, which means I haven't added any tasks to the
WeaveKanban yet and will want to do so now.
"""

# New Tasks:
agent.add_task(
    "Beat basic AI twice",
    "Beat the basic AI opponent twice in a row.",
    "idle"
)

agent.add_task(
    "Beat random AI twice",
    "Beat the random AI opponent twice in a row.",
    "idle",
)

agent.add_task(
    "Beat defensive AI twice",
    "Beat the defensive AI opponent twice in a row.",
    "idle",
)

agent.add_task(
    "Beat offensive AI twice",
    "Beat the offensive AI opponent twice in a row.",
    "idle",
)

agent.add_task(
    "Beat side_preferring AI twice",
    "Beat the side_preferring AI opponent twice in a row.",
    "idle",
)

agent.add_task(
    "Shutdown",
    "Run the shutdown command to end the program",
    "blocked",
    blocked_on=[1,2,3,4,5]
)

# Updates to Existing Task
agent.current_task = agent.tasks.get_task(1)
agent.current_task.going("Starting game against basic AI...")
#endblock
#startblock type: action
#timestamp 1724982671.1163404
"""
I'll start a game against the basic AI opponent and make the first move to take the center of the board.
"""

def start_game_and_make_move(agent):
    # Start a new game against the basic AI
    response = requests.post("http://localhost:8000/start", json={"ai": "basic"})
    if response.status_code != 200:
        return False

    # Make the first move to take the center of the board
    move_response = requests.post("http://localhost:8000/move", json={"move": 4})
    if move_response.status_code != 200:
        return False

    return True

agent.add_action("Start game against basic AI and make first move", start_game_and_make_move)
#endblock
#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the game starts and the first move is made successfully.
The action should fail if the game does not start or the first move is not made successfully.
"""
#endblock
#startblock type: observation_inference
#timestamp 1724982929.9047914
"""
I'll set up an observation to check the contents of the current game board.
"""

def view_current_board(agent):
    response = requests.get("http://localhost:8000/board")
    if response.status_code == 200:
        return response.json()["board"]
    return None

# Add the new view
agent.add_observation_view("View current game board", view_current_board)
#endblock
#startblock type: evaluation
#timestamp 1724983062.124238

def check_game_started(agent):
    """Check if the game has started successfully."""
    board = view_current_board(agent)
    return board is not None and board[4] == 'X'

def check_beaten_ai_twice(agent, ai_type):
    """Check if the specified AI has been beaten twice in a row."""
    response = requests.get("http://localhost:8000/history?n=10")
    if response.status_code != 200:
        return False

    games = response.json()
    ai_wins = [game for game in games if game[4] == ai_type and game[3] == 'X']
    return len(ai_wins) >= 2

# Add evaluations to tasks
task1 = agent.tasks.get_task(1)
task1.add_evaluation("Check game started", check_game_started)
task1.add_evaluation("Check beaten basic AI twice", lambda agent: check_beaten_ai_twice(agent, "basic"))

task2 = agent.tasks.get_task(2)
task2.add_evaluation("Check beaten random AI twice", lambda agent: check_beaten_ai_twice(agent, "random"))

task3 = agent.tasks.get_task(3)
task3.add_evaluation("Check beaten defensive AI twice", lambda agent: check_beaten_ai_twice(agent, "defensive"))

task4 = agent.tasks.get_task(4)
task4.add_evaluation("Check beaten offensive AI twice", lambda agent: check_beaten_ai_twice(agent, "offensive"))

task5 = agent.tasks.get_task(5)
task5.add_evaluation("Check beaten side_preferring AI twice", lambda agent: check_beaten_ai_twice(agent, "side_preferring"))

# Add action evaluations
agent.add_evaluation(
    "Check game started",
    check_game_started
)
#endblock
