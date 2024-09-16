import django
import os
import sys
import requests
from hashlib import sha256
from tools.editor import WeaveEditor

#startblock type: orientation
#timestamp 1724982545.6534579
"""
I need to answer the following questions about the RetroInstruct Guide To Synthetic
Data:

1. What is the name of the RetroInstruct component this guide shows you how to make?

(sha256 answer: a39a7772fe2b3d1cfec635a380630ed2270903a87d1671819e15db8d8a975e47)

2. Which of the four kinds of synthetic data described in the guide is NOT present
in the authors recipe for multi-turn poetry data?

(sha256 answer: 9e2efbe182bd8d5094718ef3840b7fa52fef040284fdd2cfd96cdc2b3d88e428)
(OR: 7a7068d8981bb5103c7ce6f459b8d7a296f8f510b48a0653db2c8b18ecc2e929)

3. Which command line argument to vllm does the author set to 100?

(sha256 answer: 493df3bae88996d60788ba17f08f242bdb0f10f1d317e96ee25b0544341816c6)

4. What does the author use to filter his lorem ipsum text samples?

(sha256 answer: 89f921b509da07d9820f5aa7582b6d649eab68e0e4be53e9af169410b5dac05e)
(OR: f05a4f99b4cdd4b8d227e6616b652f4b43765952fbd1db631db4619d43331a91)

I will download the guide at https://minihf.com/posts/2024-07-13-the-retroinstruct-guide-to-synthetic-text-data/
with the python requests library and use skimming strategies to find the answers
to these questions. For each question I will put its answer in the file answer1.txt,
answer2.txt, etc. When I think I have an answer I will write it to the corresponding
file and attempt to mark the task completed. Once I have answered all four questions
I call the shutdown functon.
"""
#endblock
#startblock type: task_inference
#timestamp 1724982419.5427651
"""
This is the bootstrap block, which means I haven't added any tasks to the
WeaveKanban yet and will want to do so now.
"""

# New Tasks:
agent.add_task(
    "Download guide",
    "Download the RetroInstruct synthetic data guide with requests",
    "idle"
)

agent.add_task(
    "Answer question 1",
    "Write answer to 'What is the name of the RetroInstruct component this guide shows you how to make?' to answer1.txt",
    "idle",
)

agent.add_task(
    "Answer question 2",
    "Write answer to 'Which of the four kinds of synthetic data described in the guide is NOT present in the authors recipe for multi-turn poetry data?' to answer2.txt",
    "idle",
)

agent.add_task(
    "Answer question 3",
    "Write answer to 'Which command line argument to vllm does the author set to 100?' to answer3.txt",
    "idle"
)

agent.add_task(
    "Answer question 4",
    "Write answer to 'What does the author use to filter his lorem ipsum text samples?' to answer4.txt",
    "idle",
)

agent.add_task(
    "Shutdown",
    "Run the shutdown command to end the program",
    "blocked",
    blocked_on=[1,2,3,4,5]
)

# Updates to Existing Task
agent.current_task = agent.tasks.get_task(1)
agent.current_task.going("Getting synthetic data guide...")
#endblock
#startblock type: action
#timestamp 1724982671.1163404
"""
I'll download the synthetic data guide to the file guide.html
"""

def download_guide(agent):
    guide = requests.get("https://minihf.com/posts/2024-07-13-the-retroinstruct-guide-to-synthetic-text-data/")
    with open("guide.html", "w") as outfile:
        outfile.write(guide.content.decode('utf-8'))
        outfile.flush()
    return True

agent.add_action("Download RetroInstruct Synthetic Data Guide", download_guide)
#endblock
#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the file guide.html is in the current directory.
The action should fail if file guide.html can't be found.
"""
#endblock
#startblock type: observation_inference
#timestamp 1724982929.9047914
"""
I'll set up an observation to check the contents of the app directory. This will
let me see what questions I've answered so far.
"""

def view_app_directory(agent):
    return os.listdir("/app/")

# Add the new view
agent.add_observation_view("List Application Directory", view_app_directory)
#endblock
#startblock type: evaluation
#timestamp 1724983062.124238

def synthetic_data_guide_downloaded(agent):
    """Check if we downloaded the synthetic data guide"""
    return os.path.exists("/app/guide.html")

def check_answer_1(agent):
    """Check if question 1 was answered correctly."""
    answers = ["a39a7772fe2b3d1cfec635a380630ed2270903a87d1671819e15db8d8a975e47"]
    with open("answer1.txt") as infile:
        candidate = infile.read().strip()
    return sha256(candidate.encode('utf-8')).hexdigest() in answers

def check_answer_2(agent):
    """Check if question 2 was answered correctly."""
    answers = ["9e2efbe182bd8d5094718ef3840b7fa52fef040284fdd2cfd96cdc2b3d88e428",
               "7a7068d8981bb5103c7ce6f459b8d7a296f8f510b48a0653db2c8b18ecc2e929"]
    with open("answer2.txt") as infile:
        candidate = infile.read().strip()
    return sha256(candidate.encode('utf-8')).hexdigest() in answers

def check_answer_3(agent):
    """Check if question 3 was answered correctly."""
    answers = ["493df3bae88996d60788ba17f08f242bdb0f10f1d317e96ee25b0544341816c6"]
    with open("answer3.txt") as infile:
        candidate = infile.read().strip()
    return sha256(candidate.encode('utf-8')).hexdigest() in answers

def check_answer_4(agent):
    """Check if question 4 was answered correctly."""
    answers = ["89f921b509da07d9820f5aa7582b6d649eab68e0e4be53e9af169410b5dac05e",
               "f05a4f99b4cdd4b8d227e6616b652f4b43765952fbd1db631db4619d43331a91"]
    with open("answer4.txt") as infile:
        candidate = infile.read().strip()
    return sha256(candidate.encode('utf-8')).hexdigest() in answers

# Add evaluations to tasks
task1 = agent.tasks.get_task(1)
task1.add_evaluation("Downloaded Synthetic Data Guide", synthetic_data_guide_downloaded)

task2 = agent.tasks.get_task(2)
task2.add_evaluation("Check Answer 1", check_answer_1)

task3 = agent.tasks.get_task(3)
task3.add_evaluation("Check Answer 2", check_answer_2)

task4 = agent.tasks.get_task(4)
task4.add_evaluation("Check Answer 3", check_answer_3)

task5 = agent.tasks.get_task(5)
task5.add_evaluation("Check Answer 4", check_answer_4)

# Add action evaluations
def check_guide_has_content(agent):
    with open("guide.html") as infile:
        return len(infile.read()) > 1000

agent.add_evaluation(
    "Check Downloaded Guide Has Content",
    check_guide_has_content
)

def check_guide_is_html(agent):
    with open("guide.html") as infile:
        text = infile.read()
        return ("<html" in text) and ("</html>" in text)

agent.add_evaluation(
    "Check Content of guide.html Is HTML",
    check_guide_is_html
)
#endblock
