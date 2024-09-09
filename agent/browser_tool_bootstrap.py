from tools.editor import WeaveEditor

#startblock type: orientation
#timestamp 1724982545.6534579
"""
The task for the agent is to create a Python library that allows it to search and browse the web using Selenium. The library should:

1. Inspect the editor code in tools/editor.py using the editor.
2. Design a similar interface for navigating web pages.
3. Implement this interface under tools/browser.py.
4. Include a feature that lets it get search results from DuckDuckGo quickly as title, url, page description/excerpt triplets through a method like .search().

Let's start by setting up the tasks on the Kanban board and writing an action to open the editor code.
"""
#endblock
#startblock type: task_inference
#timestamp 1724982419.5427651
# Current Task (which will get printed by the task-reminder block)
# ID  : 1
# Title: Inspect the editor code in tools/editor.py
# Description: Open and inspect the editor code in tools/editor.py using the editor.
# Status: idle

# New Tasks:
agent.add_task(
    "Inspect the editor code in tools/editor.py",
    "Open and inspect the editor code in tools/editor.py using the editor.",
    "idle"
)

agent.add_task(
    "Design a similar interface for navigating web pages",
    "Design a similar interface for navigating web pages based on the editor code.",
    "blocked",
    blocked_on=[1,]
)

agent.add_task(
    "Implement the interface under tools/browser.py",
    "Implement the designed interface under tools/browser.py.",
    "blocked",
    blocked_on=[2,]
)

agent.add_task(
    "Include a search feature for DuckDuckGo",
    "Include a feature that lets it get search results from DuckDuckGo quickly as title, url, page description/excerpt triplets through a method like .search().",
    "blocked",
    blocked_on=[3,]
)

# Root Task:
agent.add_task(
    "Create a Python library for web browsing and searching",
    """
    I'm writing a tool that allows the agent to search and browse the web using Selenium. The tool should:

    1. Inspect the editor code in tools/editor.py using the editor.
    2. Design a similar interface for navigating web pages.
    3. Implement this interface under tools/browser.py.
    4. Include a feature that lets it get search results from DuckDuckGo quickly as title, url, page description/excerpt triplets through a method like .search().
    """,
    "blocked",
    blocked_on=[1, 2, 3, 4]
)

# Updates to Existing Task
agent.current_task = agent.tasks.get_task(1)
agent.current_task.going("Inspecting the editor code in tools/editor.py...")

agent.current_task = agent.tasks.get_task(1)
#endblock
#startblock type: action
#timestamp 1724982671.1163404
"""I'll open the editor on the editor code so we can look at the interface
design and methods."""

def open_editor_code(agent):
    editor = WeaveEditor(agent, "tools/editor.py")
    # Access the editor from the agent tools attribute
    editor = agent.tools[0]
    # Use the up, down, start, end, goto and find methods to navigate the file
    editor.start()
    editor.find("class WeaveEditor")
    # Replace text between two lines with edit(self, start_line, end_line, new_text)
    # and append(self, text)
    # When you are done with the editor you can .close() it
    return True

agent.add_action("Open editor.py code", open_editor_code)
#endblock
#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the editor code in tools/editor.py is opened and the class WeaveEditor is found.
The action should fail if the file is not opened or the class is not found.
"""
#endblock
#startblock type: observation_inference
#timestamp 1725697638.8825483
"""I should open an observation monitor for the tools directory as well so that
I can keep track of what files are in it."""

def monitor_tools_directory(agent):
    return os.listdir("/app/tools")

agent.add_observation_view("Monitor files in /app/tools/", monitor_tools_directory)
#endblock
#startblock type: evaluation
#timestamp 1724983062.124238

def check_editor_code_opened(agent):
    return bool(agent.tools)

agent.add_evaluation(
    "Check Editor Code Opened",
    check_editor_code_opened
)
#endblock
