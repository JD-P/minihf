import os
import libtmux
import time
from datetime import datetime
import re

class WeaveZombieGame:
    """A tmux wrapper for the zombie survival game"""
    def __init__(self, agent):
        self.agent = agent
        self.agent.tools["zombie_game"] = self
        self.observation_view = {
            "type": "observation",
            "title": "ZombieSurvival",
            "callback": self.render
        }
        self.agent.add_observation_view("ZombieSurvival", self.render, tool="zombie_game")

        # Tmux setup
        self.server = libtmux.Server()
        self.session = self.server.new_session(session_name="zombie_session", kill_session=True)
        self.window = self.session.new_window(window_name="zombie_window", attach=True)
        self.window.resize(height=24, width=80)
        self.pane = self.window.split_window(attach=True, size="100%")
        
        self.pane.send_keys("python3 /app/tools/zombie_game.py")
        
        # State tracking
        self.last_state = {
            "day": 1,
            "health": 100,
            "hunger": 70,
            "location": "abandoned_store",
            "inventory": {}
        }
        self.command_history = []

    def render(self, agent):
        """Render the game state for the agent"""
        pane_lines = []  # Initialize empty as fallback
        try:
            # Capture pane safely
            pane_lines = self.pane.capture_pane()
            parsed_state = self._parse_game_state(pane_lines)
            if "root@" in "\n".join(pane_lines):
                pane_lines.append("GAME OVER - The game has ended. "
                                  "Please restart it for another round with "
                                  'send_keys("python3 /app/tools/zombie_game.py"')

            # Build observation text using parsed state
            obs = pane_lines

            # Add recent command history
            obs.extend(
                f"[{cmd['time']}] {cmd['command']} => {cmd['result']}"
                for cmd in self.command_history[-3:]
            )

            # Add action suggestions
            obs.extend([
                "\nPOSSIBLE ACTIONS:",
                "- scavenge (find resources)",
                "- move <location> (change areas)",
                "- craft <item> (create tools)",
                "- eat (consume food)",
                "- use barricade (defend at night)"
            ])

            return "\n".join(obs)

        except Exception as e:
            # Safely handle game-end state
            end_game_content = "\n".join(pane_lines)  # Get last 10 lines
            return (
                f"GAME ENDED - FINAL STATE:\n{end_game_content}\n"
                f"Error during rendering: {str(e)}"
            )

    def _parse_game_state(self, pane_lines):
        """Safer state parsing with game-end detection"""
        state = self.last_state.copy()

        # Check for game over screen first
        if any("GAME OVER" in line for line in pane_lines):
            raise RuntimeError("Game has ended")

        # Use safer inventory parsing
        inventory_line = next((l for l in pane_lines if "Inventory:" in l), "")
        try:
            state["inventory"] = ast.literal_eval(
                inventory_line.split("Inventory:")[-1].strip()
            )
        except:
            state["inventory"] = {}

        # Find current status line using regex
        status_pattern = r"DAY (\d+) \| Health: (\d+) \| Hunger: (\d+)\s+Location: (\w+)"
        for line in reversed(pane_lines):  # Search bottom-up
            match = re.match(status_pattern, line)
            if match:
                state.update({
                    "day": int(match.group(1)),
                    "health": int(match.group(2)),
                    "hunger": int(match.group(3)),
                    "location": match.group(4).lower()
                })
                break

        # Parse connections
        connections_line = next((l for l in pane_lines if "Connections:" in l), "")
        state["connections"] = [c.strip().lower() 
                              for c in connections_line.split(":")[-1].split(",")
                              if c.strip()]

        self.last_state = state
        return state

    def send_command(self, command):
        """Send a command to the game"""
        try:
            # Send command and wait for processing
            self.pane.send_keys(command + "\n")
            time.sleep(0.5)  # Allow game state to update
            
            # Capture command result
            pane_content = self.pane.capture_pane()
            result = self._parse_command_result(pane_content)
            
            # Record in history
            self.command_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "command": command,
                "result": result
            })
            
            return True
        except Exception as e:
            print(f"Command failed: {str(e)}")
            return False

    def _parse_command_result(self, pane_content):
        """Extract the result of the last command"""
        # Look for messages between last command and status line
        reversed_lines = list(reversed(pane_content))
        result = []
        
        for line in reversed_lines:
            if re.match(r"DAY \d+ \| Health:", line):
                break
            if line.strip() and not line.startswith(">"):
                result.append(line.strip())
                
        return " | ".join(reversed(result[-3:]))  # Return last 3 relevant lines

    def close(self):
        """Clean up tmux session"""
        self.session.kill_session()
        del self.agent.tools["zombie_game"]
        self.agent.remove_observation_view(self.observation_view)

# Example integration
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
    game = WeaveZombieGame(agent)
    
    # Sample interaction
    game.send_command("move forest")
    game.send_command("scavenge")
    game.send_command("craft barricade")
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        game.close()
