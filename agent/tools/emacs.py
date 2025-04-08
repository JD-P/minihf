import os
import libtmux
import time
from datetime import datetime

class WeaveEmacs:
    """A wrapper for Emacs using libtmux to control the editor."""
    def __init__(self, agent):
        """Bind tool to weave-agent and set up Emacs."""
        self.agent = agent
        self.agent.tools["emacs"] = self
        self.observation_view = {
            "type": "observation",
            "title": "WeaveEmacs",
            "callback": self.render
        }
        self.agent.add_observation_view("WeaveEmacs", self.render, tool="emacs")

        # Initialize the tmux session and pane
        self.server = libtmux.Server()
        self.session = self.server.new_session(session_name="emacs_session", kill_session=True)
        self.window = self.session.new_window(window_name="emacs_window", attach=True)
        self.window.resize(height=24, width=80)
        self.pane = self.window.split_window(attach=True, size="100%")

        # Start Emacs in the tmux pane
        self.pane.send_keys('emacs -nw\n')

    def render(self, agent):
        """Render the current Emacs buffer state."""
        rendered_text = "'''Emacs Editor State:\n"
        try:
            pane_content = self.pane.capture_pane(start=0, end="-")
        except Exception as e:
            return f"Error capturing Emacs buffer: {str(e)}"
        
        if not pane_content:
            return "Emacs buffer is empty. Start editing to see content."
        
        # Show last 20 lines of the buffer
        buffer_excerpt = "\n".join(pane_content[-20:])
        rendered_text += f"Current Buffer Excerpt:\n{buffer_excerpt}\n\n"

        command_cheat_sheet = (
            "Emacs Key Bindings Cheat Sheet:\n"
            "  C-x C-f : Open/create file\n"
            "  C-x C-s : Save current buffer\n"
            "  C-x C-w : Save buffer as...\n"
            "  C-x C-c : Exit Emacs\n"
            "  C-g     : Cancel current command\n"
            "  C-s     : Search forward\n"
            "  C-r     : Search backward\n"
            "  C-a     : Beginning of line\n"
            "  C-e     : End of line\n"
            "  C-n     : Next line\n"
            "  C-p     : Previous line\n"
            "  C-v     : Scroll down\n"
            "  M-v     : Scroll up\n"
            "  M-x     : Execute extended command\n"
            "  C-k     : Kill line\n"
            "  C-y     : Yank (paste)\n"
            "  C-space : Set mark\n"
            "  C-w     : Kill region\n"
            "  M-w     : Copy region\n"
            "  C-_     : Undo\n"
            "  C-x u   : Undo\n"
            "  C-x d   : Open directory\n"
            "  C-x b   : Switch buffer\n"
        )
        rendered_text += command_cheat_sheet
        rendered_text += "\n'''"
        return rendered_text

    def send_keys(self, command):
        """Send a keyboard command to Emacs."""
        self.pane.send_keys(command, enter=False)
        time.sleep(0.2)  # Allow time for buffer updates

    def send_command(self, command):
        """Alias to send commands to Emacs."""
        self.send_keys(command)

    def close(self):
        """Close the Emacs session."""
        self.session.kill_session()
        del self.agent.tools["emacs"]
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
    emacs = WeaveEmacs(agent)

    # Simulate basic Emacs usage
    time.sleep(2)  # Wait for Emacs to start
    emacs.send_command('C-x C-f test.txt')
    emacs.send_command('Hello World')
    emacs.send_command('C-x C-s')
    emacs.send_command('C-x C-c')

    try:
        while True:
            pass
    except KeyboardInterrupt:
        emacs.close()
