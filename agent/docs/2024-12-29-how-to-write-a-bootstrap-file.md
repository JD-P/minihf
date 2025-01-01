---
layout: blogpost
title: How To Write A Bootstrap File
date: 2024-12-31
author: John David Pressman
tags: [doc, blog]
---

**META:** The short answer is that you copy [this prompt](https://gist.github.com/JD-P/79d2b5c38e72e986bb203c5c49386d9a)
into [Mistral-large](https://chat.mistral.ai/) or another intelligent long context
language model, replace the task instructions given at the top with the task you
want a bootstrap file for, and then edit the result until it's correct. The rest
of this article is the instructions the model is given so it knows how to do that.
How will you know when the bootstrap file is correct? What even *is* a weave-agent
bootstrap file? Read on and you will learn.

A weave-agent bootstrap file is a manually written set of python code blocks
that set up the agent to perform a specific task. It appears in the agent's
context window as the first tick of the event loop, but is actually written
before the agent starts. It serves three purposes:

1) Providing a few shot prompt to get the underlying language model started
writing weave-agent traces.

2) Framing the task for the agent with the opportunity to provide examples of
API calls, import tools and libraries that will be necessary to perform the
task. Basically do the parts you know need to be done at the start with a script
so the agent always succeeds at them.

3) Show the agent itself doing the task, so that it observes itself having
already made the decision to do the task and putting effort towards it with a
successful first outcome. This premises the rest of the completions on the idea
that the agent is already doing the task and that it is competent to do the
task.

Mechanically how the bootstrap file works is that a mock subagent is made:

```
    # Mock bootstrap agent so we can run the callbacks in bootstrap file
    self = agent.subagent(
        "bootstrap",
        None,
        "Bootstrap the weave-agent",
        {},
        args.budget,
                          
    )
```

The bootstrap file code is then executed, this defines a series of callbacks on
the mock agent in the standard places they would appear during a real tick of
the event loop:

```
    with open(args.bootstrap) as infile:
        # Bootstrap block
        bootstrap_block = {
            'type': 'bootstrap',
            'body': infile.read()
        }
        self.add_block(bootstrap_block)
        exec(bootstrap_block["body"])
```

The callbacks are then retrieved from their expected locations on the subagent
and executed. Afterwards the mock subagent is deleted.

```
    def run_bootstrap_callbacks(subagent):
        """Run bootstrap callbacks in function to avoid contaminating global scope."""
        # Run action callback
        action_result = subagent.current_tick.action["callback"](subagent)

        # Run evaluation callbacks
        evaluation_results = []
        for evaluation in subagent.current_tick.evaluations:
            result = evaluation["callback"](subagent)
            evaluation_results.append((evaluation['title'], result))

        outcomes =  []
        outcomes += [(subagent.current_tick.action["title"],action_result),]
        outcomes += evaluation_results

        # Add outcome block
        outcome_block = {
            'type': 'outcome',
            'table': outcomes
        }
        subagent.add_block(outcome_block)
        subagent.current_tick.outcome = outcome_block

    run_bootstrap_callbacks(self)
    # Clean up mock bootstrap agent
    del(self)
```

The reason it's done this way is to be polite to the weave-agent. Since the
bootstrap file is presented to it as being part of the trace, we try to keep
the underlying generative process and semantics of the bootstrap block as close
to a real weave-agent event loop as possible.

### Structure And Requirements Of A Bootstrap File

The weave-agent bootstrap format has undergone major changes as the framework
has evolved. The [single tic tac toe game](https://github.com/JD-P/minihf/blob/main/agent/bootstraps/tictactoe_single_bootstrap.py)
and [vigenere cipher](https://github.com/JD-P/minihf/blob/main/agent/bootstraps/vigenere_bootstrap.py)
files are current so you can use them to few shot prompt an instruction model
into writing you something with the right format and then you can edit it.
To help your language model with that I will label the different parts of the
single tic tac toe game bootstrap file and explain how to write them.

The first part of a bootstrap file, the preamble, isn't labeled. It's the series
of import statements and setup at the top before the orientation block.

```
import requests
import json
import threading
import time
from http.server import HTTPServer
from bootstraps.tictactoe_server import TicTacToeHandler

# Start the server in a separate thread
server = HTTPServer(('localhost', 8000), TicTacToeHandler)
server_thread = threading.Thread(target=server.serve_forever)
server_thread.daemon = True
server_thread.start()
time.sleep(1)  # Give the server some time to start

# Start a new game against the basic AI
response = requests.post("http://localhost:8000/start", json={"ai": "basic"})
assert response.status_code == 200
```

This is both normal python code and optional. I suggest importing any libraries
you expect the agent to use during the task so it doesn't have to import them
itself during action blocks. You can also influence how the model chooses to solve
the task by importing some libraries vs. others. In the case of tic tac toe we import
a separate HTTP based tic tac toe game server whose code and unit tests are kept
in other files. This is done because the bootstrap file is put into the agent
trace and we don't want to fill up the model's context window with unnecessary
text.

The second part of a bootstrap file is the orientation. Right now this is just
a chain of thought [in a modified Hermes format](https://gist.github.com/JD-P/47e0d4aa2b255a38e2eddeddd761a971).
The intention of Hermes is to bring together chain of thought prompting and
simulated multi-agent debate prompting, both of which are known to improve AI
reasoning independently.

```
#startblock type: orientation
#timestamp 1724982545.6534579
"""
WEAVER [P: EXPECTATION], I'm in a game of tic tac toe against a dumb opponent.
I want to win the game and then return to parent. The game is being played
on a HTTP server served on localhost 8000.

WEAVER [P: CLARIFICATION], How do I make a move?

WEAVER [P: EXPOSITION], You make a move using the /move endpoint and the requests
library. For example: `requests.post("http://localhost:8000/move", json={"move": 4})`
lets us take the center of the board.

WEAVER [P: CLARIFICATION], How do I get the board state?

WEAVER [P: EXPOSITION], You use the /board endpoint, which returns a JSON in this
format: {"board": ["O", " ", " ", " ", "X", " ", " ", " ", " "]} Keep in mind that
the empty spaces on the board are a space string rather than none or empty string.

WEAVER [P: RATIONAL], And I get the first move, so I can take the center?

WEAVER [P: EXPOSITION], Yes, we can take the center of the board.

WEAVER [P: CONCLUSION], Alright then I will use the following strategy:
1. Make a move to take the center of the board since that's the best opening move.
2. Continue making moves based on the current state of the board.

I will use the /board endpoint to observe the current state of the board.
Once I have won or run out of time I will return to parent.
"""
#endblock
```

The basic syntax is the name of the system that the epistemic postures (denominated with p:)
are speaking as part of. Since this is a monologue there is only the WEAVER speaker,
but future iterations might have a USER or ADMIN that can speak to the model
during its reasoning chain. To write this kind of chain of thought figure out
which emotional-epistemic-posture you would use or want the agent to use or that
would speak next and then write down its statement. There's no fixed set of
postures but the current ones mentioned in the weave-agent source that are used
to prompt the model are:

```
            + "# what to do in light of this. Some postures weave-agent\n"
            + "# has include:\n"
            + "#\n"
            + "# WEAVER [P: EXPECTATION], I analyze whether the expectation\n"
            + "# was met or not by the observable results of the previous\n"
            + "# action.\n"
            + "#\n"
            + "# WEAVER [P: HYPOTHESIS], I enumerate different hypothesis\n"
            + "# and point out ways we could gain more information about\n"
            + "# which of them is true.\n"
            + "#\n"
            + "# WEAVER [P: RATIONAL], I focus on inferences we can make\n"
            + "# by employing first principles reasoning or logical\n"
            + "# extrapolation from well known mental models and premises.\n"
            + "#\n"
            + "# WEAVER [P: EMPIRICISM], I focus on inferences we can make\n"
            + "# by paying attention to sensory observations and concrete\n"
            + "# examples. I have a habit of pointing out when an extrapolation\n"
            + "# from RATIONAL is contradicted by an observable phenomenon\n"
            + "# or piece of evidence from the world. We then reconcile\n"
            + "# the contradiction together.\n"
            + "#\n"
            + "# WEAVER [P: RATIONAL], We do actually discuss things by the\n"
            + "# way.\n"
            + "#\n"
            + "# WEAVER [P: EMPIRICISM], As you could have inferred from the\n"
            + "# description of the Morpheus format above this conversation,\n" 
            + "# yes. Let's continue.\n"
            + "#\n"
            + "# WEAVER [P: ARBITER], I coordinate the discussion and help\n"
            + "# resolve disputes that arise between weave-agent's personas.\n"
            + "# I'm especially likely to appear if things are starting to\n"
            + "# get overly rude or derail.\n"
            + "#\n"
            + "# WEAVER [P: ARBITER], By the way a posture can talk twice in\n"
            + "# a row if it has meaningfully separate thoughts about\n"
            + "# something and it would make the most ergonomic sense to\n"
            + "# separate them.\n"
            + "#\n"
            + "# WEAVER [P: RATIONAL-2], Postures can also talk to themselves\n"
            + "# if their thought comes from the same emotional-cognitive place.\n"
            + "#\n"
            + "# WEAVER [P: RATIONAL-1], Yeah but I don't have anything to say\n"
            + "# to myself right now so introduce the next guy.\n"
            + "#\n"
            + "# WEAVER [P: CONCLUSION], I appear at the end of the discussion\n"
            + "# to write the concluding block outlining our next steps as a\n"
            + "# bullet point list. Speaking of which, it's time to get started!\n"
```

Including some of these postures in your chain of debate would help give models
trained on the resulting traces more regular structure to latch onto.

The next part of the bootstrap file is taking an action based on what was reasoned
in the orientation stage.

```
#startblock type: action
#timestamp 1724982671.1163404
"""
I'll set up the main agent to play the rest of the game and make the first move to 
take the center of the board.
"""

def main_agent_and_move(subagent):
    def check_beaten_ai(subagent, ai_type):
        """Check if the specified AI has been beaten twice in a row."""
        response = requests.get("http://localhost:8000/history?n=10")
        if response.status_code != 200:
            return False

        games = response.json()
        ai_wins = [game for game in games if game[4] == ai_type and game[3] == 'X']
        return len(ai_wins) >= 1

    # Avoid using lambda because we need named unit test to satisfy return_to_caller
    def victory(subagent):
        return check_beaten_ai(subagent, "basic_ai")
    
    schema = {"victory":"boolean"}
    main_agent = agent.subagent("main", None, "Beat basic AI at Tic-Tac-Toe", schema, args.budget)
    main_agent.task.add_evaluation("Check beaten basic AI", victory)

    def view_current_board(subagent):
        response = requests.get("http://localhost:8000/board")
        if response.status_code == 200:
            return response.json()["board"]
        return None

    # Add the new view to the main agent instead of self since it's about to be executed
    main_agent.add_observation_view("View current game board", view_current_board)
    
    # Make the first move to take the center of the board
    move_response = requests.post("http://localhost:8000/move", json={"move": 4})
    if move_response.status_code != 200:
        return False
    return True

self.add_action("Set up main agent and make first move", main_agent_and_move)
#endblock
```

This action block is where we start to diverge from the structure of a normal
weave-agent tick. The action block in the bootstrap file must define the root
subagent, `main` and it needs to give main its task unit tests and return schema.
The schema is [a JSON schema](https://json-schema.org/) that defines what
information the subagent is expected to return to its caller. Because the observation
view we want to start the agent with is being bound to the "main" subagent rather
than the mock bootstrap subagent we omit the `observation-inference` stage and
instead bind the observation callback as part of the action callback along with
the unit tests like `victory`. The action callback here also makes the first
move as the bootstrap subagent. This structure is admittedly a little confusing
and that's because it wasn't really designed, but what fell out from making
changes to an earlier design. In the future I'd like to make a class based syntax
for defining the subagents but right now this is what you get. To help you make
sense of it as it is, `main_agent_and_move` is the action callback, inside of
this we define the `main` subagent which is a necessary thing for the bootstrap
file to do. The `main` subagent is given a task unit test called `victory` which
is a wrapper around a more general function `check_beaten_ai`. The subagent is
also given an observation callback `view_current_board` and then finally an action
is taken to advance the board state and show the weave-agent how to make its moves.

The rest of the tick is the normal expectation and evaluation stages.

```
#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the game starts and the first move is made successfully.
The action should fail if the game does not start or the first move is not made successfully.
"""
#endblock
#startblock type: evaluation
#timestamp 1724983062.124238

def check_game_started(subagent):
    """Check if the game has started successfully."""
    response = requests.get("http://localhost:8000/board")
    if response.status_code == 200:
        board = response.json()["board"]
    else:
        return False
    return board[4] == 'X'

# Add action evaluations
self.add_evaluation(
    "Check game started",
    check_game_started
)
#endblock
```


### Task Verification Patterns

There are useful design patterns for writing task evaluations which expand what
you can verify. Here are some I've discovered so far while writing bootstrap files:

#### External State Checking

The most common evaluation pattern is writing programs that pull some state from
the external environment and check it with traditional boolean logic.

This simple example constructs the filepath for a Wikipedia page that's expected
to have been downloaded and then uses the python `os` library to check that the
file exists.

```
def check_wiki_markup_downloaded(agent):
    page_title = "Example_page"
    file_path = f"{page_title}.wiki"
    return os.path.exists(file_path)
```

#### Asserting API's That Don't Yet Exist

A more advanced use of external state checks is to write checks that explicitly or
implicitly assume conditions like the existence of a certain API that does not yet
exist by performing a simple operation with that conjectural API.

For example here is an evaluation callback from [Django Agent Interface Bootstrap](https://github.com/JD-P/minihf/blob/main/agent/bootstraps/agent_django_interface_bootstrap.py)
that tries to detect whether the agent has created a download button or not:

```
def evaluate_download_button(agent):
    """Check if the download button is created"""
    with open("/app/weave_agent_webapp/main/views.py", "r") as file:
        content = file.read()
        return "def download_button(request):" in content
```

A close observer will notice that this doesn't actually check whether the button
exists in the page, or if it functions to let you download anything. That's because
this evaluation was written when the weave-agent was at a much earlier stage of
development where putting down anything structurally related to the objective
would be an achievement (still sort of true, really). What this evaluation checks
is whether there exists a view to handle collating the data when a user presses
the download button. The parameter that accepts the calling subagent is also named
'agent' because weave-agent used to only have one layer of agent loop.

Here's another example from [Browser Tool Bootstrap](https://github.com/JD-P/minihf/blob/main/agent/bootstraps/browser_tool_bootstrap.py) that determines whether
the DuckDuckGo search is working:

```
def browser_ddg_search(agent):
    from tools.browser import WeaveBrowser
    browser = WeaveBrowser(agent, "https://minihf.com/")
    results = browser.search("stable diffusion wikipedia")
    # Results should be NamedTuples
    assert results[0].url == "https://en.wikipedia.org/wiki/Stable_Diffusion"
    assert results[0].title == "Stable Diffusion - Wikipedia"
```

The idea in both of these evaluations is to induce the weave-agent to write
a particular API or interface by showing it a unit test it has to pass which
expects that API or interface to exist. The second example shows we can encode
write down particular results when certain data structures are accessed or function
calls are made. This can be a useful way to give more specific structure to
requests like "write me a library that does X" or "write me a tool to Y".

#### Solution Hashing

One common pattern is that you want the weave-agent to perform a task like break
a cipher or solve a math equation where writing a unit test that checks for the
answer string would give away the solution. Writing a unit test that performs a
series of operations to determine if the answer is correct would give away the
procedure the agent is supposed to figure out to arrive at the solution. This
can be solved by hashing the solution and then comparing to the hex representation
of the answer hash. The following example comes from [Vigenere Cipher
Bootstrap](https://github.com/JD-P/minihf/blob/main/agent/bootstraps/vigenere_bootstrap.py):

```
    def check_correct_plaintext(subagent):
        """Check if we've found the correct plaintext."""
        with open("solution.txt") as infile:
            candidate = sha256(infile.read().strip().encode('utf-8')).hexdigest()
        return candidate == 'a528933eabc4772c9a99436fbd5fc592d7ffc16f6bd94ae76aafcbbe32e0cdc3'
```

Here the plaintext has been hashed out-of-band and then inserted into the bootstrap
file. The procedure to check its correctness is to hash a particular expected
filename, `solution.txt` and see if its SHA256 hexdigest matches that of the
plaintext known to the bootstrap creator. In order for this procedure to work
`solution.txt` needs to contain the plaintext, which is not known to the agent
at the start of the task. It should break the cipher, put the plaintext in that
file, and then the unit test will pass.

Another place where I've used this pattern is in [Skim RetroInstruct Data Guide
Bootstrap](https://github.com/JD-P/minihf/blob/main/agent/bootstraps/skim_retroinstruct_data_guide.py)
where I use it to have the model answer reading comprehension questions about the
[RetroInstruct Guide To Synthetic Text Data](https://minihf.com/posts/2024-07-13-the-retroinstruct-guide-to-synthetic-text-data/):

```
def check_answer_1(agent):
    """Check if question 1 was answered correctly."""
    answers = ["a39a7772fe2b3d1cfec635a380630ed2270903a87d1671819e15db8d8a975e47"]
    with open("answer1.txt") as infile:
        candidate = infile.read().strip()
    return sha256(candidate.encode('utf-8')).hexdigest() in answers
```

#### Weave Evaluator

Often we want to evaluate a subjective quality of some sensory data. For example
it would be silly to expect we could have the weave-agent write a good short story
using boolean logic and string matching as the evaluation method. Luckily the
agent framework provides a library to call a logic evaluator to ask yes-no
questions. This example comes from [Dome 39 SciFi Bootstrap](https://github.com/JD-P/minihf/blob/main/agent/bootstraps/dome39_scifi_bootstrap.py)
which tries to write a short story about a Mars colony:

```
def evaluate_story_beginning(agent):
    question = "Does the story have a clear beginning?"
    with open("scifi.txt", "r") as file:
        story = file.read()
    score_prompt_fn = make_simple_score_prompt(question)
    score = simple_evaluate_outputs(score_prompt_fn, story)[0].item()
    return score >= 3
```

A similar evaluation callback is used in [Haunted Mansion Bootstrap](https://github.com/JD-P/minihf/blob/main/agent/bootstraps/haunted_mansion_bootstrap.py):

```
def evaluate_story_engagement(agent):
    question = "Is the story engaging and interesting?"
    with open("horror.txt", "r") as file:
        story = file.read()
    score_prompt_fn = make_simple_score_prompt(question)
    score = simple_evaluate_outputs(score_prompt_fn, story)[0].item()
    if score >= 3:
        return True
    else:
        return score
```