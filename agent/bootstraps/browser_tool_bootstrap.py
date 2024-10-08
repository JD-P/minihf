import selenium
from tools.editor import WeaveEditor

#startblock type: orientation
#timestamp 1724982545.6534579
"""
The task for the agent is to create a Python library that allows it to search and browse the web using Selenium. The library should:

1. Inspect the editor code in tools/editor.py using WeaveEditor.
2. Design and implement a similar interface under tools/browser.py using WeaveEditor.
3. Include a feature that lets it get search results from DuckDuckGo quickly as title, url, page description/excerpt triplets through a method like .search().

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
    "Implement the interface in the tools/browser.py file using the WeaveEditor",
    "Implement the designed interface under tools/browser.py.",
    "blocked",
    blocked_on=[1,]
)
    
agent.add_task(
    "Add a search feature for DuckDuckGo using the WeaveEditor",
    "Add a feature that lets it get search results from DuckDuckGo quickly as title, url, page description/excerpt triplets through a method like .search().",
    "blocked",
    blocked_on=[2,]
)
    
# Root Task:
agent.add_task(
    "Create a Python library for web browsing and searching",
    """
    I'm writing a tool that allows me to search and browse the web using 
    Selenium. The tool should implement an interface similar to tools/editor.py
    but with the up() and down() methods working more like elinks and going down
    n links rather than n lines. The tool should also allow me to search the Internet
    by parsing DuckDuckGo results into title, url, page description/excerpt triplets.
    """,
    "blocked",
    blocked_on=[1, 2, 3]
)

# Updates to Existing Task
agent.current_task = agent.tasks.get_task(1)
agent.current_task.going("Inspecting the editor code in tools/editor.py...")

agent.current_task = agent.tasks.get_task(1)
#endblock
#startblock type: action
#timestamp 1724982671.1163404
"""I'll open the editor on the editor code so we can look at the interface
design and methods. I will also open a second editor for tools/browser.py """

def open_editor_code(agent):
    editor = WeaveEditor(agent, "tools/editor.py")
    # Access the editor from the agent tools attribute
    editor = agent.tools[0]
    editor.window_size = 100
    # Use the up, down, start, end, goto and find methods to navigate the file
    editor.start()
    editor.find("class WeaveEditor")
    # Replace text between two lines with edit(self, start_line, end_line, new_text)
    # and append(self, text)
    editor2 = WeaveEditor(agent, "tools/browser.py")
    editor2.append("class WeaveBrowser:\n    pass")
    # When you are done with the editor you can .close() it
    editor2.close()
    return True

agent.add_action("Open editor.py code", open_editor_code)
#endblock
#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the editor code in tools/editor.py is opened and the class WeaveEditor is found.
The action should fail if the file is not opened or the class is not found.
I also expect to find a file at tools/browser.py with the start of a WeaveBrowser 
class in it.
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

# Task Evaluations

def browser_py_methods(agent):
    from tools.browser import WeaveBrowser
    assert hasattr(WeaveBrowser, "start")
    assert hasattr(WeaveBrowser, "end")
    assert hasattr(WeaveBrowser, "up")
    assert hasattr(WeaveBrowser, "down")
    assert hasattr(WeaveBrowser, "open")
    assert hasattr(WeaveBrowser, "close")
    assert hasattr(WeaveBrowser, "find")
    assert hasattr(WeaveBrowser, "render")

def browser_py_interface(agent):
    from tools.browser import WeaveBrowser
    browser = WeaveBrowser(agent, "https://minihf.com/")
    assert browser.render(agent).startswith("#URL: https://minihf.com/ (10 URLs total)#\n")
    assert "To make this possible MiniHF provides several powerful features:" in browser.render(agent)
    browser.close()
    
def browser_ddg_search(agent):
    from tools.browser import WeaveBrowser
    browser = WeaveBrowser(agent, "https://minihf.com/")
    results = browser.search("stable diffusion wikipedia")
    # Results should be NamedTuples
    assert results[0].url == "https://en.wikipedia.org/wiki/Stable_Diffusion"
    assert results[0].title == "Stable Diffusion - Wikipedia"

task2 = agent.tasks.get_task(2)
task2.add_evaluation("browser.py WeaveBrowser class exists with correct methods",
                     browser_py_methods)
task2.add_evaluation("browser.py interface is like editor.py interface",
                     browser_py_interface)

task3 = agent.tasks.get_task(3)
task3.add_evaluation("browser has working DuckDuckGo search",
                     browser_ddg_search)
    
# Action Evaluations

def check_editor_code_opened(agent):
    return bool(agent.tools)

def check_browser_py_exists(agent):
    return os.path.exists("/app/tools/browser.py")

def check_browser_py_class(agent):
    with open("/app/tools/browser.py") as infile:
        return infile.read() == "class WeaveBrowser:\n"

agent.add_evaluation(
    "Check Editor Code Opened",
    check_editor_code_opened
)

agent.add_evaluation(
    "Check browser.py Exists",
    check_browser_py_exists
)

agent.add_evaluation(
    "Check browser.py Has WeaveBrowser Class Start In It",
    check_browser_py_class
)
#endblock
