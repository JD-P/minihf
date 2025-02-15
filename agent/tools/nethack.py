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
        self.pane = self.window.split_window(attach=True)

        # Start Nethack in the tmux pane
        self.pane.send_keys('/usr/games/nethack\n')

        # Init screen and move history
        self.moves = []

    def render(self, agent):
        """Render the Nethack game display."""
        rendered_text = "'''Nethack Game History (Last 5 Keys Sent):\n"
        if not self.moves:
            return "You haven't made any moves yet. Please send some keys to see the game."
        for i, move in enumerate(self.moves[-5:]):
            screen, keys, timestamp = move
            screen = "\n".join(screen)
            hh_ss = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            rendered_text += f"<{hh_ss}>\n{screen}\nKeys: \"{keys}\"\n"
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
            "  F + Direction keys ( fight a monster )"
            "  f ( fire ammunition from quiver )"
            "  Q ( select ammunition )"
            "  C-t ( teleport, if you can )"
        )
        rendered_text += command_cheat_sheet
        rendered_text += "\n'''"
        return rendered_text

    def send_keys(self, command):
        """Send a keyboard command to the Nethack game."""
        self.pane.send_keys(command)
        pane_content = self.pane.capture_pane(start=0, end="-")
        self.moves.append((pane_content, command, time.time()))

    def close(self):
        """Close the Nethack session."""
        self.session.kill_session()
        del self.agent.tools["nethack"]
        self.agent.remove_observation_view(self.observation_view)

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
