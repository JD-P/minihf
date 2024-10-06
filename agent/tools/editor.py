import re

class WeaveEditor:
    """An LLM text editor similar to the one described in the
       swe-agent paper (https://arxiv.org/abs/2405.15793)."""
    def __init__(self, agent, filepath):
        """Bind tool to weave-agent and set up editor."""
        self.agent = agent
        self.filepath = filepath
        self.agent.tools[f"editor-{filepath}"] = self
        self.editor_observation_view = {"type":"observation",
                                        "title":f"WeaveEditor ({filepath})",
                                        "callback":self.render}
        self.agent.add_observation_view(f"WeaveEditor ({filepath})", self.render)

        # Initialize the editor state
        self.current_line = 0
        self.window_size = 10
        self.open(filepath)

    def open(self, filepath):
        self.file_content = self.load_file(filepath)
        self.filepath = filepath
        
    def close(self):
        self.agent.tools.remove(self)
        self.agent.remove_observation_view(self.editor_observation_view)

    def load_file(self, filepath):
        """Load the file content into a list of lines."""
        try:
            with open(filepath, 'r') as file:
                content = file.read()
                return split_lines_with_custom_newlines(content, "\n\r\x0b\x0c")
                
        except FileNotFoundError:
            with open(filepath, 'w') as file:
                file.write("")
                file.flush()
                return ["",]

    def save_file(self, filepath):
        """Save the file content from the list of lines."""
        with open(filepath, 'w') as file:
            file.writelines(self.file_content)
        self.file_content = self.load_file(filepath)
        self.filepath = filepath

    def render(self, agent):
        """Render the text block based on the current line number position and the window size."""
        total_lines = len(self.file_content)
        start_line = max(0, self.current_line - self.window_size // 2)
        end_line = min(total_lines, self.current_line + self.window_size // 2)

        # Create the rendered text block
        rendered_text = f"#File: {self.filepath} ({total_lines} lines total)#\n"
        rendered_text += f"'''({start_line} lines above)\n"
        for i in range(start_line, end_line):
            rendered_text += f"{i+1} {self.file_content[i]}"
        rendered_text += f"\n({total_lines - end_line} lines below)\n'''"
        return rendered_text

    def edit(self, start_line, end_line, new_text):
        """Replace the text between two lines with the text given in a separate argument."""
        start_line -= 1
        end_line -= 1
        if start_line < 0  or start_line > end_line:
            raise ValueError("Invalid line range")
        if end_line >= len(self.file_content):
            end_line = len(self.file_content)
        self.file_content[start_line:end_line+1] = new_text.splitlines(keepends=True)
        self.save_file(self.filepath)

    def unidiff_edit(self, unidiff):
        edits = parse_diff(unidiff)
        for start_line, end_line, new_text in edits:
            self.edit(start_line, end_line, new_text)
        
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

def parse_diff_header(line):
    # Updated regex to handle optional count
    hunk_header_regex = re.compile(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
    match = hunk_header_regex.match(line)
    if match:
        start_line_old = int(match.group(1))
        count_old = int(match.group(2)) if match.group(2) else 1
        start_line_new = int(match.group(3))
        count_new = int(match.group(4)) if match.group(4) else 1
        return start_line_old, count_old, start_line_new, count_new
    return None

def process_hunk(lines, start_line):
    edits = []
    offset = 0
    local_offset = 0
    current_line = start_line
    context_above = None
    context_below = None
    changed = False

    line_stack = []
    if not lines[0].startswith(" "):
        context_above = ''
        line_stack.append('')
    for line in lines:
        # Apply effects of + or - first in case last line
        if line.startswith('-'):
            current_line += 1
            local_offset -= 1
            changed = True
        elif line.startswith('+'):
            local_offset += 1
            # current_line += 1
            changed = True

        if line.startswith(' ') or line == lines[-1]:
            if context_above is None:
                context_above = line[1:]
                start_line = current_line
                line_stack.append(context_above)
                current_line += 1
            else:
                context_below = line[1:]
                line_stack.append(context_below)
                if changed:
                    edits.append(
                        (start_line,
                         current_line,
                         ''.join(line_stack))
                    )
                offset += local_offset
                current_line += local_offset
                start_line = current_line
                local_offset = 0
                changed = False
                line_stack = []
                context_above = line[1:]
                line_stack.append(context_above)
                current_line += 1
        elif line.startswith('+'):
            line_stack.append(line[1:])
 
    assert (len([line for line in lines if line.startswith("+")])
             - len([line for line in lines if line.startswith("-")])) == offset
    return edits, offset

def parse_diff(unidiff):
    if type(unidiff) == str:
        lines = unidiff.splitlines()
    elif type(unidiff) == list:
        lines = unidiff
    else:
        raise ValueError("unidiff must be string or splitlines list")
    edits = []
    i = 0

    offset = 0
    header_count = len([line for line in lines if line.startswith('@@')])
    headers = []
    while i < len(lines):
        line = lines[i]
        if line.startswith('@@'):
            header_info = parse_diff_header(line)
            headers.append(header_info)
            if header_info:
                start_line_old, count_old, start_line_new, count_new = header_info
                hunk_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith('@@'):
                    assert lines[i][0] in (' ', '+', '-')
                    hunk_lines.append(lines[i])
                    i += 1
                start_line_old += offset
                start_line_new += offset
                new_edits, hunk_offset = process_hunk(hunk_lines, start_line_old)
                offset += hunk_offset
                edits.extend(new_edits)
        if not line.startswith('@@'):
            i += 1

    assert header_count == len(headers)
    return edits

def split_lines_with_custom_newlines(content, newline_chars):
    """
    Splits the lines in a file based on a custom set of newline characters.

    :param file_path: Path to the file to be read.
    :param newline_chars: A set of characters to be considered as newlines.
    :return: A list of lines split based on the custom newline characters.
    """
    lines = []
    current_line = []

    for char in content:
        current_line.append(char)
        if char in newline_chars:
            lines.append(''.join(current_line))
            current_line = []
    # Add the last line if there is any remaining content
    if current_line:
        lines.append(''.join(current_line))

    return lines
