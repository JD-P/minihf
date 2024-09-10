import requests
from functools import partial
from tools.editor import WeaveEditor

#startblock type: orientation
#timestamp 1724982545.6534579
"""
The task for the agent is to extract and validate citations from a Wikipedia page.
The tool should:

1. Get the Wiki markup for a Wikipedia page.
2. Parse and extract the citations.
3. Associate the citations with the claims in the article they are meant to back up.
4. Download the web resources associated with the citations that have a URL.
5. Use various search strategies to find the most relevant potential passages or artifacts on the page which could back up the claim.
6. Compare the most relevant sections found with the claim in the article associated with that citation.
7. Record the verdict of whether the citation is valid or not in a tally, aggregate, or ledger which can be examined by a human being.

Let's start by setting up the tasks on the Kanban board and writing an action to download the Wiki markup.
"""
#endblock
#startblock type: task_inference
#timestamp 1724982419.5427651
# Current Task (which will get printed by the task-reminder block)
# ID  : 1
# Title: Get the Wiki markup for a Wikipedia page
# Description: Download and store the Wiki markup for a given Wikipedia page.
# Status: idle

# New Tasks:
agent.add_task(
    "Get the Wiki markup for a Wikipedia page",
    "Download and store the Wiki markup for a given Wikipedia page.",
    "idle"
)

agent.add_task(
    "Parse and extract the citations",
    "Parse the Wiki markup to extract citations.",
    "blocked",
    blocked_on=[1,]
)

agent.add_task(
    "Associate citations with claims",
    "Associate the extracted citations with the claims in the article they are meant to back up.",
    "blocked",
    blocked_on=[2,]
)

agent.add_task(
    "Download web resources associated with citations",
    "Download the web resources associated with the citations that have a URL.",
    "blocked",
    blocked_on=[3,]
)

agent.add_task(
    "Search for relevant passages in the resources",
    "Use various search strategies to find the most relevant potential passages or artifacts on the page which could back up the claim.",
    "blocked",
    blocked_on=[4,]
)

agent.add_task(
    "Compare relevant sections with claims",
    "Compare the most relevant sections found with the claim in the article associated with that citation.",
    "blocked",
    blocked_on=[5,]
)

agent.add_task(
    "Record the verdict of citation validity",
    "Record the verdict of whether the citation is valid or not in a tally, aggregate, or ledger which can be examined by a human being.",
    "blocked",
    blocked_on=[6,]
)

# Root Task:
agent.add_task(
    "Extract and validate citations from a Wikipedia page",
    """
    I'm writing a tool that extracts a list of citations from a Wikipedia page and checks the ones with URLs for correctness. The tool should:

    1. Get the Wiki markup for a Wikipedia page
    2. Parse and extract the citations
    3. Associate the citations with the claims in the article they are meant to back up
    4. Download the web resources associated with the citations that have a URL
    5. Use various search strategies to find the most relevant potential passages or artifacts on the page which could back up the claim
    6. Compare the most relevant sections found with the claim in the article associated with that citation.
    7. Record the verdict of whether the citation is valid or not in a tally, aggregate, or ledger which can be examined by a human being.
    """,
    "blocked",
    blocked_on=[1, 2, 3, 4, 5, 6, 7]
)

# Updates to Existing Task
agent.current_task = agent.tasks.get_task(1)
agent.current_task.going("Downloading the Wiki markup for a Wikipedia page...")

agent.current_task = agent.tasks.get_task(1)
#endblock
#startblock type: action
#timestamp 1724982671.1163404

def download_wiki_markup(page_title):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&titles={page_title}"
    response = requests.get(url)
    data = response.json()
    pages = data['query']['pages']
    page_id = next(iter(pages))
    markup = pages[page_id]['revisions'][0]['*']
    with open(f"{page_title}.wiki", "w") as file:
        file.write(markup)

    WeaveEditor(agent, "Linux.wiki")

    # Access the editor from the agent tools attribute
    editor = agent.tools[0]
    editor.window_size = 50

    # Use the up, down, start, end, goto and find methods to navigate the file
    editor.end()
    editor.start()
    editor.find("cite web")

    # Replace text between two lines with edit(self, start_line, end_line, new_text)
    # and append(self, text)
    # When you are done with the editor you can .close() it
    return True

def download_wiki_markup_action(agent):
    return download_wiki_markup("Linux")

agent.add_action("Download Wiki markup", download_wiki_markup_action)
#endblock
#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the Wiki markup for the given page is downloaded and saved to a file.
The action should fail if the request to the Wikipedia API fails or if the file is not created.
"""
#endblock
#startblock type: observation_inference
#timestamp 1725697638.8825483
"""I've already opened the WeaveEditor on the Wiki markup so I'll pass for now."""
pass
#endblock
#startblock type: evaluation
#timestamp 1724983062.124238

def check_wiki_markup_downloaded(agent):
    page_title = "Example_page"
    file_path = f"{page_title}.wiki"
    return os.path.exists(file_path)

def check_editor_on_agent(agent):
    from tools.editor import WeaveEditor
    return type(agent.tools[0]) == WeaveEditor

agent.add_evaluation(
    "Check We Downloaded Wiki Markup File",
    check_wiki_markup_downloaded
)

agent.add_evaluation(
    "Check editor is equipped on agent",
    check_editor_on_agent
)
#endblock
