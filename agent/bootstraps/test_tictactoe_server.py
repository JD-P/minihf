import os
import random
import unittest
import requests
import threading
import time
from http.server import HTTPServer

if os.path.exists('tictactoe.db'):
    os.remove('tictactoe.db')

from tictactoe_server import get_winner
    
def start_game(ai):
    response = requests.post('http://localhost:8000/start', json={'ai': ai})
    return response

def make_move(move):
    response = requests.post('http://localhost:8000/move', json={'move': move})
    return response

def finish_game(board):
    while True:
        available_moves = [position[0] for position in enumerate(board) if position[1] == " "]
        if not available_moves:
            return
        response = make_move(random.choice(available_moves))
        try:
            board = response.json()['board']
        except KeyError:
            if "winner" in response.json():
                return
            elif response.json()['error'] == "No game in progress.":
                return
            else:
                raise ValueError

EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'
        
class TestGetWinner(unittest.TestCase):
    def test_winner_first_row(self):
        board = [PLAYER_X, PLAYER_X, PLAYER_X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
        self.assertEqual(get_winner(board), PLAYER_X)

    def test_winner_second_row(self):
        board = [EMPTY, EMPTY, EMPTY, PLAYER_O, PLAYER_O, PLAYER_O, EMPTY, EMPTY, EMPTY]
        self.assertEqual(get_winner(board), PLAYER_O)

    def test_winner_third_row(self):
        board = [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, PLAYER_X, PLAYER_X, PLAYER_X]
        self.assertEqual(get_winner(board), PLAYER_X)

    def test_winner_first_column(self):
        board = [PLAYER_O, EMPTY, EMPTY, PLAYER_O, EMPTY, EMPTY, PLAYER_O, EMPTY, EMPTY]
        self.assertEqual(get_winner(board), PLAYER_O)

    def test_winner_second_column(self):
        board = [EMPTY, PLAYER_X, EMPTY, EMPTY, PLAYER_X, EMPTY, EMPTY, PLAYER_X, EMPTY]
        self.assertEqual(get_winner(board), PLAYER_X)

    def test_winner_third_column(self):
        board = [EMPTY, EMPTY, PLAYER_O, EMPTY, EMPTY, PLAYER_O, EMPTY, EMPTY, PLAYER_O]
        self.assertEqual(get_winner(board), PLAYER_O)

    def test_winner_first_diagonal(self):
        board = [PLAYER_X, EMPTY, EMPTY, EMPTY, PLAYER_X, EMPTY, EMPTY, EMPTY, PLAYER_X]
        self.assertEqual(get_winner(board), PLAYER_X)

    def test_winner_second_diagonal(self):
        board = [EMPTY, EMPTY, PLAYER_O, EMPTY, PLAYER_O, EMPTY, PLAYER_O, EMPTY, EMPTY]
        self.assertEqual(get_winner(board), PLAYER_O)

    def test_no_winner(self):
        board = [PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_O, PLAYER_X, PLAYER_O]
        self.assertEqual(get_winner(board), None)
        
class TestTicTacToeServer(unittest.TestCase):
    def setUp(self):
        # Start the server in a separate thread
        from tictactoe_server import TicTacToeHandler
        self.server = HTTPServer(('localhost', 8000), TicTacToeHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        time.sleep(1)  # Give the server some time to start

    def tearDown(self):
        # Stop the server
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()

    def test_basic_ai(self):
        start_game('basic')
        response = make_move(0)
        self.assertIn('board', response.json())
        self.assertEqual(response.json()['board'][0], 'X')
        ai_move = next(i for i, spot in enumerate(response.json()['board']) if spot == 'O')
        self.assertEqual(ai_move, 1)  # Basic AI should take the first available spot

    def test_defensive_ai(self):
        start_game('defensive')
        response = make_move(0)
        self.assertIn('board', response.json())
        self.assertEqual(response.json()['board'][0], 'X')
        ai_move = next(i for i, spot in enumerate(response.json()['board']) if spot == 'O')
        self.assertIn(ai_move, [1, 3, 4, 5, 7])  # Defensive AI should block potential winning moves

    def test_offensive_ai(self):
        start_game('offensive')
        response = make_move(0)
        self.assertIn('board', response.json())
        self.assertEqual(response.json()['board'][0], 'X')
        ai_move = next(i for i, spot in enumerate(response.json()['board']) if spot == 'O')
        self.assertIn(ai_move, [1, 2, 3, 4, 5, 6, 7, 8])  # Offensive AI should try to win

    def test_side_preferring_ai(self):
        start_game('side_preferring')
        response = make_move(0)
        self.assertIn('board', response.json())
        self.assertEqual(response.json()['board'][0], 'X')
        ai_move = next(i for i, spot in enumerate(response.json()['board']) if spot == 'O')
        self.assertIn(ai_move, [1, 3, 5, 7])  # Side-preferring AI should prefer side spots

    def test_start_game_in_progress(self):
        start_game('basic')
        response = requests.post('http://localhost:8000/start', json={'ai': 'random'})
        self.assertEqual(response.status_code, 409)

    def test_game_history(self):
        for _ in range(3):
            start_game('basic')
            board = make_move(0).json()['board']
            finish_game(board)
            
        response = requests.get('http://localhost:8000/history?n=10')
        self.assertEqual(response.status_code, 200)
        games = response.json()
        self.assertEqual(len(games), 3)

    def test_player_move(self):
        finish_game([" "] * 9)
        start_game('basic')
        response = make_move(0)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['board'][0], 'X')

        response = make_move(2)
        self.assertEqual(response.json()['board'][2], 'X')

if __name__ == '__main__':
    unittest.main()
