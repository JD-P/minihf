import os
import libtmux
import time
from datetime import datetime

class WeaveNethack:
    """A wrapper for Nethack using libtmux to control the game."""
    def __init__(self, agent):
        """Bind tool to weave-agent and set up Nethack."""
        self.agent = agent
        self.agent.tools["nethack"] = self
        self.observation_view = {
            "type": "observation",
            "title": "WeaveNethack",
            "callback": self.render
        }
        self.agent.add_observation_view("WeaveNethack", self.render, tool="nethack")

        # Initialize the tmux session and pane
        self.server = libtmux.Server()
        self.session = self.server.new_session(session_name="nethack_session", kill_session=True)
        self.window = self.session.new_window(window_name="nethack_window", attach=True)
        self.window.resize(height=24, width=80)
        self.pane = self.window.split_window(attach=True, size="100%")

        # Start Nethack in the tmux pane
        self.pane.send_keys('/usr/games/nethack\n')

        # Init screen and move history
        self.moves = []

    def render(self, agent):
        """Render the Nethack game display."""
        rendered_text = "'''Nethack Game History (Last 5 Keys Sequences Sent):\n"
        if not self.moves:
            return "You haven't made any moves yet. Please send some keys to see the game."
        for i, move in enumerate(self.moves[-5:]):
            screen, keys, timestamp = move
            screen = "\n".join(screen)
            hh_ss = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            rendered_text += f"History Frame #{i} <{int(timestamp)}>\n"
            if move == self.moves[-1]:
                rendered_text += "(CURRENT FRAME)\n"
            rendered_text += f"{screen}\nKeys: \"{keys}\"\n"
            # Make it clear whether the PC has moved
            rendered_text += f"Player Character Position: {screen.find('@')}"
            try:
                if screen.find('@') != "\n".join(self.moves[-6+i][0]).find('@'):
                    rendered_text += " ( Changed )\n"
                else:
                    rendered_text += "\n"
            # Don't track changes if there aren't at least 6 moves yet
            except IndexError:
                rendered_text += "\n"
        rendered_text += "Notable Objects In Current Frame:\n"
        onscreen_objects = self.get_objects()
        if len(onscreen_objects) == 1:
            rendered_text += "{No notable objects on screen}\n"
        else:
            for key, value in self.get_objects().items():
                rendered_text += f"{key}: {value}\n"
        command_cheat_sheet = (
            "Key Command Cheat Sheet:\n"
            "  y ( move northwest )\n"
            "  k ( move north )\n"
            "  u ( move northeast )\n"
            "  h ( move west )\n"
            "  l ( move east )\n"
            "  b ( move southwest )\n"
            "  j ( move south )\n"
            "  n ( move southeast )\n"
            "  Escape ( cancel an action )\n"
            "  C-a ( redo last command )\n"
            "  C-c ( quit the game )\n"
            "  S ( save the game and exit )\n"
            "  & ( get hint about what a key does )\n"
            "  ? ( show the help menu )\n"
            "  h ( show the help menu )\n"
            "  / ( navigate to a map character to learn what sort of thing it is )\n"
            "  ; ( navigate to a map character to learn what it is )\n"
            "  < ( go up stairs )\n"
            "  > ( go down stairs )\n"
            "  j ( jump )\n"
            "  z ( zap a wand )\n"
            "  Z ( cast a spell )\n"
            "  t ( throw )\n"
            "  s ( search for hidden doors/etc around you )\n"
            "  E ( engrave writing into the floor )\n"
            "  ^ ( show trap type of adjacent traps )\n"
            "  u ( untrap )\n"
            "  F + Direction keys ( fight a monster )\n"
            "  f ( fire ammunition from quiver )\n"
            "  Q ( select ammunition )\n"
            "  C-t ( teleport, if you can )"
        )
        rendered_text += command_cheat_sheet
        rendered_text += "\n'''"
        return rendered_text

    def send_keys(self, command):
        """Send a keyboard command to the Nethack game."""
        self.pane.send_keys(command)
        time.sleep(0.1)
        pane_content = self.pane.capture_pane(start=0, end="-")
        self.moves.append((pane_content, command, time.time()))

    def send_command(self, command):
        """Alias to send a keyboard command to the NetHack game which Qwen keeps
           hallucinating exists."""
        self.send_keys(command)

    def simple_move(self, coordinates):
        x_move, y_move = coordinates
        keys = ""
        if x_move > 0:
            keys += "l" * x_move
        elif x_move < 0:
            keys += "h" * (-1 * x_move)
        if y_move > 0:
            keys += "k" * y_move
        elif y_move < 0:
            keys += "j" * (-1 * y_move)
        if "go_downstairs" in locals():
            keys += ">"
        self.send_keys(keys)
        
    def close(self):
        """Close the Nethack session."""
        self.session.kill_session()
        del self.agent.tools["nethack"]
        self.agent.remove_observation_view(self.observation_view)

    # Add these symbol mappings at the class level
    FEATURE_SYMBOLS = {
        '-': 'horizontal wall',
        '|': 'vertical wall',
        '+': 'door',
        '#': 'corridor',
        '.': 'floor',
        '<': 'up staircase',
        '>': 'down staircase',
        '^': 'trap',
        '_': 'altar',
        '=': 'throne',
        '"': 'pool',
    }

    MONSTER_SYMBOLS = {
        # Uppercase monsters (hostile)
        'A': 'giant ant', 'B': 'bat', 'C': 'centaur', 'D': 'dragon',
        'E': 'flaming sphere', 'F': 'violet fungi', 'G': 'gnome',
        # Lowercase monsters (mostly peaceful)
        'a': 'grid bug', 'b': 'bat', 'c': 'centipede', 'd': 'dog',
        'e': 'energy vortex', 'f': 'fox', 'g': 'goblin',
        # Add more as needed from wiki
    }

    ITEM_SYMBOLS = {
        ')': 'weapon', '(': 'projectile', '[': 'armor', ']': 'ring',
        '/': 'wand', '?': 'scroll', '!': 'potion', '=': 'amulet',
        '"': 'spellbook', '$': 'gold piece', '_': 'tool'
    }

    def get_objects(self):
        """Return parsed game objects with relative coordinates"""
        if not self.moves:
            return {}
        
        current_screen = self.moves[-1][0]
        parsed = self._parse_screen(current_screen)
        return self._generate_object_dict(parsed)

    MAP_HEIGHT = 20  # NetHack's standard map height in 80x24 terminal

    def _parse_screen(self, screen_lines):
        """Parse screen contents into raw object positions"""
        player_pos = None
        objects = {}
        map_area = screen_lines[:self.MAP_HEIGHT]

        # Find player position within map area
        for y, line in enumerate(map_area):
            if '@' in line:
                x = line.index('@')
                player_pos = (x, y)
                break
        
        if not player_pos:
            return {}

        # Process all map area characters
        px, py = player_pos
        for y, line in enumerate(map_area):
            for x, char in enumerate(line):
                if char == '@' or char == ' ':  # Skip player and empty spaces
                    continue
                
                # Calculate relative coordinates (north=positive Y)
                rel_x = x - px
                rel_y = py - y  # Inverted Y-axis

                # Classify characters using symbol mappings
                obj_type = None
                if char in self.FEATURE_SYMBOLS:
                    obj_type = self.FEATURE_SYMBOLS[char]
                elif char in self.MONSTER_SYMBOLS:
                    obj_type = self.MONSTER_SYMBOLS[char]
                elif char in self.ITEM_SYMBOLS:
                    obj_type = self.ITEM_SYMBOLS[char]
                
                if obj_type:
                    if obj_type not in objects:
                        objects[obj_type] = []
                    objects[obj_type].append((rel_x, rel_y))

        return {'player': (px, py), 'objects': objects}
    
    def _generate_object_dict(self, parsed):
        """Generate the final object dictionary with shortcut entries"""
        if not parsed:
            return {}
        
        result = {'player': (0, 0)}  # Always relative to player
        px, py = parsed['player']
        objects = parsed['objects']

        # Process exits (doors and corridors)
        exits = []
        for exit_type in ['door', 'corridor']:
            exits += objects.get(exit_type, [])
        
        if exits:
            result['nearest-exits'] = self._find_nearest(exits)
            result['northmost-exit'] = self._find_extreme(exits, 'y', max)
            result['southmost-exit'] = self._find_extreme(exits, 'y', min)
            result['eastmost-exit'] = self._find_extreme(exits, 'x', max)
            result['westmost-exit'] = self._find_extreme(exits, 'x', min)

        # Process stairs
        stairs = []
        for stair_type in ['up staircase', 'down staircase']:
            stairs += objects.get(stair_type, [])
        
        if stairs:
            result['nearest-stairs'] = self._find_nearest(stairs)
            result['northmost-stairs'] = self._find_extreme(stairs, 'y', max)

        # Add walls and other features as lists
        for feature in ['horizontal wall', 'vertical wall', 'floor']:
            if feature in objects:
                result[feature] = objects[feature]

        # Add monsters and items as lists
        for obj_type in list(self.MONSTER_SYMBOLS.values()) + list(self.ITEM_SYMBOLS.values()):
            if obj_type in objects:
                result[obj_type] = objects[obj_type]

        # Filter out empty entries
        return {k: v for k, v in result.items() if (isinstance(v, list) and len(v) > 0) or (not isinstance(v, list))}

    def _find_nearest(self, positions):
        """Find position with smallest Euclidean distance"""
        positions.sort(key=lambda p: p[0]**2 + p[1]**2)
        return positions

    def _find_extreme(self, positions, axis, func):
        """Find extreme position in specified axis"""
        idx = 0 if axis == 'x' else 1
        return func(positions, key=lambda p: p[idx])

    # ... existing methods ...

# Example usage
if __name__ == "__main__":
    class DummyAgent:
        def __init__(self):
            self.tools = {}
            self.observation_views = []

        def add_observation_view(self, title, callback):
            self.observation_views.append((title, callback))

        def remove_observation_view(self, view):
            self.observation_views.remove(view)

    agent = DummyAgent()
    nethack = WeaveNethack(agent)

    # Simulate sending commands to Nethack
    time.sleep(2)  # Wait for Nethack to start
    nethack.send_command('h')  # Move left
    time.sleep(1)
    nethack.send_command('j')  # Move down
    time.sleep(1)
    nethack.send_command('k')  # Move up
    time.sleep(1)
    nethack.send_command('l')  # Move right

    # Keep the main process running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        nethack.close()
