import http.server
import json
import sqlite3
from urllib.parse import urlparse, parse_qs
import random

# Tic-tac-toe board representation and AI

EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'

def initial_board():
    return [EMPTY for _ in range(9)]

def get_winner(board):
    winning_combos = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for combo in winning_combos:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != EMPTY:
            return board[combo[0]]
    return None

def basic_ai(board):
    # Simple AI that takes the first available spot
    for i, spot in enumerate(board):
        if spot == EMPTY:
            return i
    return None

def random_ai(board):
    # AI that takes a random available spot
    available_spots = [i for i, spot in enumerate(board) if spot == EMPTY]
    return random.choice(available_spots) if available_spots else None

def defensive_ai(board):
    # AI that prioritizes blocking the player's winning moves
    for combo in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
        if board[combo[0]] == board[combo[1]] == PLAYER_X and board[combo[2]] == EMPTY:
            return combo[2]
        if board[combo[0]] == board[combo[2]] == PLAYER_X and board[combo[1]] == EMPTY:
            return combo[1]
        if board[combo[1]] == board[combo[2]] == PLAYER_X and board[combo[0]] == EMPTY:
            return combo[0]
    return random_ai(board)

def offensive_ai(board):
    # AI that prioritizes its own winning moves
    for combo in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
        if board[combo[0]] == board[combo[1]] == PLAYER_O and board[combo[2]] == EMPTY:
            return combo[2]
        if board[combo[0]] == board[combo[2]] == PLAYER_O and board[combo[1]] == EMPTY:
            return combo[1]
        if board[combo[1]] == board[combo[2]] == PLAYER_O and board[combo[0]] == EMPTY:
            return combo[0]
    return random_ai(board)

def side_preferring_ai(board):
    # AI that prefers taking side spots
    preferred_spots = [1, 3, 5, 7]
    available_spots = [i for i in preferred_spots if board[i] == EMPTY]
    return random.choice(available_spots) if available_spots else random_ai(board)

# Setup global game state

def setup_game():
    global game_state
    game_state = {
        'current_board': initial_board(),
        'current_player': PLAYER_X,
        'ai_strategy': None,
        'move_count': 0,
        'game_in_progress': False
    }

setup_game()

# Database setup

def database_setup():
    conn = sqlite3.connect('tictactoe.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS games (id INTEGER PRIMARY KEY, first_player TEXT, moves INTEGER, winner TEXT, ai_strategy TEXT)''')
    conn.commit()
    conn.close()

database_setup()

# Request handlers

class TicTacToeHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/start':
            if game_state['game_in_progress']:
                self.send_response(409)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Game already in progress."}).encode('utf-8'))
                return

            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            ai_strategy_name = json.loads(post_data)['ai']

            ai_strategies = {
                'basic': basic_ai,
                'random': random_ai,
                'defensive': defensive_ai,
                'offensive': offensive_ai,
                'side_preferring': side_preferring_ai,
            }

            if ai_strategy_name not in ai_strategies:
                self.send_response(400)
                self.end_headers()
                return

            game_state['ai_strategy'] = ai_strategies[ai_strategy_name]
            game_state['current_board'] = initial_board()
            game_state['current_player'] = PLAYER_X
            game_state['move_count'] = 0
            game_state['game_in_progress'] = True

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'message': 'New game started!'}).encode('utf-8'))

        elif self.path == '/move':
            if not game_state['game_in_progress']:
                self.send_response(409)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No game in progress."}).encode('utf-8'))
                return

            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            move = json.loads(post_data)['move']

            if game_state['current_board'][move] != EMPTY:
                self.send_response(409)
                self.end_headers()
                self.wfile.write(json.dumps({"board": game_state['current_board'],
                                             "error": "That spot is taken"}).encode('utf-8'))
                return

            game_state['current_board'][move] = PLAYER_X
            game_state['move_count'] += 1

            winner = get_winner(game_state['current_board'])
            if winner:
                self.save_game(winner)
                response = {'winner': winner}
                game_state['game_in_progress'] = False
            else:
                ai_move = game_state['ai_strategy'](game_state['current_board'])
                if ai_move is not None:
                    game_state['current_board'][ai_move] = PLAYER_O
                    game_state['move_count'] += 1
                winner = get_winner(game_state['current_board'])
                if winner:
                    self.save_game(winner)
                    response = {'winner': winner}
                elif EMPTY not in game_state['current_board']:
                    self.save_game("TIE")
                    response = {'winner': "TIE"}
                else:
                    response = {'board': game_state['current_board']}

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

    def do_GET(self):
        if self.path.startswith('/history'):
            query_components = parse_qs(urlparse(self.path).query)
            n = int(query_components['n'][0]) if 'n' in query_components else 10

            conn = sqlite3.connect('tictactoe.db')
            c = conn.cursor()
            try:
                c.execute("SELECT * FROM games ORDER BY id DESC LIMIT ?", (n,))
                games = c.fetchall()
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
                return
            finally:
                conn.close()

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(games).encode('utf-8'))

        elif self.path == '/board':
            if not game_state['game_in_progress']:
                self.send_response(409)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No game in progress."}).encode('utf-8'))
                return

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"board": game_state['current_board']}).encode('utf-8'))

            
    def save_game(self, winner):
        conn = sqlite3.connect('tictactoe.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO games (first_player, moves, winner, ai_strategy) VALUES (?, ?, ?, ?)",
                      (PLAYER_X, game_state['move_count'], winner, game_state['ai_strategy'].__name__))
            conn.commit()
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        finally:
            conn.close()
        setup_game()

if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = http.server.HTTPServer(server_address, TicTacToeHandler)
    httpd.serve_forever()
