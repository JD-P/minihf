
# Set up the root task
root_task = {
    "type": "task",
    "title": "Extract and validate citations from a Wikipedia page",
    "priority": 0,
    "root": True,
    "description": """
    Write me a tool that extracts a list of citations from a Wikipedia page and checks the ones with URLs for correctness. The tool should:

    1. Get the Wiki markup for a Wikipedia page
    2. Parse and extract the citations
    3. Associate the citations with the claims in the article they are meant to back up
    4. Download the web resources associated with the citations that have a URL
    5. Use various search strategies to find the most relevant potential passages or artifacts on the page which could back up the claim
    6. Compare the most relevant sections found with the claim in the article associated with that citation.
    7. Record the verdict of whether the citation is valid or not in a tally, aggregate, or ledger which can be examined by a human being.
    """
}
agent.add_task(root_task)

# Set up sub-tasks
sub_tasks = [
    {
        "type": "task",
        "title": "Get the Wiki markup for a Wikipedia page",
        "priority": 1,
        "description": "Download and store the Wiki markup for a given Wikipedia page."
    },
    {
        "type": "task",
        "title": "Parse and extract the citations",
        "priority": 1,
        "description": "Parse the Wiki markup to extract citations."
    },
    {
        "type": "task",
        "title": "Associate citations with claims",
        "priority": 1,
        "description": "Associate the extracted citations with the claims in the article they are meant to back up."
    },
    {
        "type": "task",
        "title": "Download web resources associated with citations",
        "priority": 1,
        "description": "Download the web resources associated with the citations that have a URL."
    },
    {
        "type": "task",
        "title": "Search for relevant passages in the resources",
        "priority": 1,
        "description": "Use various search strategies to find the most relevant potential passages or artifacts on the page which could back up the claim."
    },
    {
        "type": "task",
        "title": "Compare relevant sections with claims",
        "priority": 1,
        "description": "Compare the most relevant sections found with the claim in the article associated with that citation."
    },
    {
        "type": "task",
        "title": "Record the verdict of citation validity",
        "priority": 1,
        "description": "Record the verdict of whether the citation is valid or not in a tally, aggregate, or ledger which can be examined by a human being."
    }
]
for task in sub_tasks:
    agent.add_task(task)

# Demonstrate one full loop of orienting, writing an action, forming an expectation,
# and writing evaluation callbacks

#startblock type: orientation
'''
The first step is to get the Wiki markup for a Wikipedia page.
We need to download and store the Wiki markup for a given Wikipedia page.
Let's start by writing an action to download the Wiki markup.
'''
#endblock
#startblock type: action
import requests

def download_wiki_markup(page_title):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&titles={page_title}"
    response = requests.get(url)
    data = response.json()
    pages = data['query']['pages']
    page_id = next(iter(pages))
    markup = pages[page_id]['revisions'][0]['*']
    with open(f"{page_title}.wiki", "w") as file:
        file.write(markup)
    return markup

download_wiki_markup("Example")
#endblock
#startblock type: expectation
'''
The action should succeed if the Wiki markup for the given page is downloaded and saved to a file.
The action should fail if the request to the Wikipedia API fails or if the file is not created.
'''
#endblock
#startblock type: evaluation

import os

def check_wiki_markup_downloaded(agent):
    page_title = "Example_page"
    file_path = f"{page_title}.wiki"
    return os.path.exists(file_path)

agent.add_evaluation({
    "type": "unit_test",
    "title":"Check We Downloaded Wiki Markup File",
    "callback": check_wiki_markup_downloaded,
})
#endblock
