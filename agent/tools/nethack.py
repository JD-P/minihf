import os
import libtmux
import time

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
        self.agent.add_observation_view("WeaveNethack", self.render)

        # Initialize the tmux session and pane
        self.server = libtmux.Server()
        self.session = self.server.new_session(session_name="nethack_session", kill_session=True)
        self.window = self.session.new_window(window_name="nethack_window", attach=True)
        self.pane = self.window.split_window(attach=True)

        # Start Nethack in the tmux pane
        self.pane.send_keys('/usr/games/nethack\n')

    def render(self, agent):
        """Render the Nethack game display."""
        pane_content = self.pane.capture_pane()
        rendered_text = "'''Nethack Game Display:\n"
        rendered_text += "\n".join(pane_content) + "\n'''"
        return rendered_text

    def send_keys(self, command):
        """Send a keyboard command to the Nethack game."""
        self.pane.send_keys(command)

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
