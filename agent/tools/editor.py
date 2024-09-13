import re

class WeaveEditor:
    """An LLM text editor similar to the one described in the
       swe-agent paper (https://arxiv.org/abs/2405.15793)."""
    def __init__(self, agent, filepath):
        """Bind tool to weave-agent and set up editor."""
        self.agent = agent
        self.filepath = filepath
        self.agent.tools.append(self)
        self.editor_observation_view = {"type":"observation",
                                        "title":f"WeaveEditor ({filepath})",
                                        "callback":self.render}
        self.agent.add_observation_view(f"WeaveEditor ({filepath})", self.render)

        # Initialize the editor state
        self.current_line = 0
        self.window_size = 10
        self.file_content = self.load_file(filepath)

    def close(self):
        self.agent.tools.remove(self)
        self.agent.remove_observation_view(self.editor_observation_view)

    def load_file(self, filepath):
        """Load the file content into a list of lines."""
        try:
            with open(filepath, 'r') as file:
                return file.readlines()
        except FileNotFoundError:
            with open(filepath, 'w') as file:
                file.write("")
                file.flush()
                return ["",]

    def save_file(self, filepath):
        """Save the file content from the list of lines."""
        with open(filepath, 'w') as file:
            file.writelines(self.file_content)

    def render(self, agent):
        """Render the text block based on the current line number position and the window size."""
        total_lines = len(self.file_content)
        start_line = max(0, self.current_line - self.window_size // 2)
        end_line = min(total_lines, self.current_line + self.window_size // 2)

        # Create the rendered text block
        rendered_text = f"#File: {self.filepath} ({total_lines} lines total)#\n"
        rendered_text += f"'''({start_line} lines above)\n"
        for i in range(start_line, end_line):
            rendered_text += f"{i+1}: {self.file_content[i]}"
        rendered_text += f"({total_lines - end_line} lines below)\n'''"
        return rendered_text

    def edit(self, start_line, end_line, new_text):
        """Replace the text between two lines with the text given in a separate argument."""
        if start_line < 0 or end_line >= len(self.file_content) or start_line > end_line:
            raise ValueError("Invalid line range")
        self.file_content[start_line:end_line+1] = new_text.splitlines(keepends=True)
        self.save_file(self.filepath)

    def append(self, text):
        """Append the given text to the end of the file."""
        self.file_content.append(text)
        self.save_file(self.filepath)

    def up(self, n):
        """Move the line position up n lines."""
        self.current_line = max(0, self.current_line - n)

    def down(self, n):
        """Move the line position down n lines."""
        self.current_line = min(len(self.file_content) - 1, self.current_line + n)

    def goto(self, line):
        """Change the line position to the given argument."""
        if line < 0 or line >= len(self.file_content):
            raise ValueError("Invalid line number")
        self.current_line = line

    def find(self, regex):
        """Move the line position to the next line after the current line position with a substring matching the given regex."""
        pattern = re.compile(regex)
        for i in range(self.current_line + 1, len(self.file_content)):
            if pattern.search(self.file_content[i]):
                self.current_line = i
                return True
        # Pattern not found
        return False

    def start(self):
        """Go to the first line in the file."""
        self.current_line = 0

    def end(self):
        """Go to the last line in the file."""
        self.current_line = len(self.file_content) - 1

    def window_size(self, n):
        """Change the display window size to n lines."""
        self.window_size = n
