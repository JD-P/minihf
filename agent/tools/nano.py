from functools import partial
import libtmux
import time
import os

class WeaveNano:
    """A wrapper for Nano using libtmux to control the editor."""
    def __init__(self, agent, filepath):
        self.agent = agent
        self.filepath = os.path.abspath(filepath)
        # Limit to one instance per file
        # This can be changed later if it wants multiple views of the same file(?)
        if f"nano-{self.filepath}" in self.agent.tools:
            editor = self.agent.tools[f"nano-{self.filepath}"]
            editor.close()
        self.agent.tools[f"nano-{self.filepath}"] = self
        # Let agent grab the last instianted nano instance with this shortcut
        self.agent.tools["nano"] = self
        self.observation_view = {
            "type": "observation",
            "title": "WeaveNano",
            "callback": self.render
        }
        # Prevent empty WeaveNano object when observation views are at capacity
        try:
            self.agent.add_observation_view("WeaveNano", self.render, tool="nano")
        except ValueError as e:
            del self.agent.tools[f"nano-{self.filepath}"]
            del self.agent.tools["nano"]
            raise ValueError("Couldn't open editor because there are too many "
                             + "observation views. Try removing some.")
            
        # Tmux setup
        self.server = libtmux.Server()
        session_name = f"nano_{self.filepath}".replace(".","")
        self.session = self.server.new_session(session_name=session_name, kill_session=True)
        self.window = self.session.new_window(window_name="nano_window", attach=True)
        self.window.resize(height=24, width=80)
        self.pane = self.window.split_window(attach=True, size="100%")

        # Start Nano
        self.pane.send_keys(f'nano {filepath}\n')
        time.sleep(1)  # Wait for Nano to start

    def render(self, agent):
        """Render the Nano editor display."""
        try:
            content = ''
            for i, line in enumerate(self.pane.capture_pane(start=0, end="-")):
                content += f"{i+1}. | {line}\n"
            content = content[:-1]
            return f"""'''Nano Editor State:
{content}
-----------------
Nano Cheat Sheet:
  Ctrl+O - Save
  Ctrl+X - Exit
  Ctrl+K - Cut line
  Ctrl+U - Paste
  Ctrl+W - Search
  Ctrl+\\ - Replace
  Ctrl+G - Help
'''"""
        except Exception as e:
            return f"Error getting Nano state: {str(e)}"
        
    def send_keys(self, command):
        """Send commands to Nano with proper timing"""
        # Special handling for control sequences
        self.pane.send_keys(command, enter=False)
        time.sleep(0.2)  # Nano needs time to process

    def send_command(self, command):
        """Alias of send_keys"""
        self.send_keys(command)

    def send_commands(self, commands):
        for command in commands:
            self.send_keys(command)

    def get_text(self):
        """Alias for render()"""
        return render(self.agent)

    def close(self):
        """Close the Nano session"""
        self.session.kill_session()
        del self.agent.tools["nano"]
        del self.agent.tools[f"nano-{self.filepath}"]
        self.agent.remove_observation_view(self.observation_view)
