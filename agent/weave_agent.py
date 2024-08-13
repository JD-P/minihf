#hint Listen Carefully: This program is self modifying code.
# It works by feeding the program text into a large language
# model and generating the next code block as an addition to a
# long python file. Each code block is marked by #startblock and
# #endblock. The code between those lines is compiled by the agent
# framework below to check it is syntactically and semantically
# valid.

# The weave-agent is basically something like a ReAct pattern https://arxiv.org/abs/2210.03629
# instantiated in the context of a long 'python file', kind of like an ipython
# notebook but without the special syntax. It takes inspiration from Voyager
# https://arxiv.org/abs/2305.16291 and Cradle https://baai-agents.github.io/Cradle/
# in that the agents actions are fully general python programs that interact with
# the computable environment rather than specialized 'tool calling' or similar
# mechanisms. This turns out to make more sense than trying to directly hook up
# the model to interfaces for a few reasons:
# 1) Writing out its actions as programs lets the model batch its actions together
# to form coherent motions rather than getting stuck on fine grained details if it
# generates its actions token by token in the moment.
# 2) These models are highly optimized for writing code whereas interacting with
# whatever interface you have is either marginal in the pretraining set or actually
# out of distribution.
# 3) Programming APIs are already well developed for basically any task you might
# want to try and automate. If it can be symbolically manipulated as text there
# probably exists a python API to interact with it. This makes the python code
# interface highly general in the same way Cradle solves the interface problems
# vision language models have by having them write out their actions as mouse +
# keyboard inputs with code.
# 4) 'A long python file' provides what Janus would call a diegetic interface.
# It is a natural frame in which basically anything is allowed to happen, while
# still framing events and recursive context switching in a way that helps ground
# the model and prevent it from getting swept up into a predictive model of
# whatever is happening. It reminds the model that it has a perspective which
# exists outside of whatever it's currently looking at.
# The weave-agent improves on previous frameworks by including easy access to logit
# evaluators and prompting the agent to check that its actions were successful
# before moving on to the next task. In order to perform a long chain of actions
# successfully it's necessary to carefully ensure each intermediate step is
# completed before moving on to the next step. For evaluations that require
# subjective judgment this can be difficult to do with traditional program logic.
# This is why the logit evaluator provided by the framework is an important
# primitive for the agent to check its work.

import os
import json
import random
import time
import types
import traceback
import requests
import torch
from argparse import ArgumentParser
from functools import partial
from tqdm import tqdm
from weave import generate_outputs_vllm, evaluate_outputs_vllm
from weave import make_score_prompt_vllm

def make_simple_score_prompt(question: str):
    """Simplify the process of making a weave evaluator question prompt maker so
    that it's just a matter of passing a question for the weave-agent."""
    template = f"<s> [INST]{{response}}\n\nAnswer yes or no and only yes or no. {question}"
    return partial(make_score_prompt_vllm, template, "[/INST]", "")

with open("python.lark") as infile:
    python_grammar = infile.read()


class Tick:
    def __init__(self, agent, index):
        self._agent = agent
        self.tick_id = index
        self.evaluations = []

    def validate(self):
        if not hasattr(self, 'orientation'):
            raise ValueError("No orientation on tick.")
        elif not hasattr(self, 'action'):
            raise ValueError("No action on tick.")
        elif type(self.action['program']) != types.FunctionType:
            raise TypeError("Tick action is not callback.")

        elif not hasattr(self, 'expectation'):
            raise ValueError("No expectation on tick.")
        elif not self.evaluations:
            raise ValueError("No evaluations on tick.")
        elif not hasattr(self, 'outcome'):
            raise ValueError("No outcome on tick.")

    def to_json(self):
        return {
            "tick_id":self.tick_id,
            "orientation":self.orientation,
            "action":repr(self.action),
            "expectation":self.expectation,
            "evaluations":repr(self.evaluations),
            "outcome":repr(self.outcome),
        }                
        
    
class WeaveAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.event_stream = []
        self.current_tick = Tick(self, 0)
        self.ticks = []
        self.current_block_index = 0
        self.reminders = []
        self.tasks = []
        self.observation_views = []
        self.cache = {}
        self.context = ""

    def shutdown(self):
        """The agent shutdown routine. This should be called when the agent's 
        root task has been resolved, the root task is deemed intractable, or the
        agent has wandered off so far it can't find its way back to the task."""
        if not os.path.exists("weave-agent-logs"):
            os.mkdir("weave-agent-logs")
        with open(f"weave-agent-logs/{round(time.time())}/log.json", "w") as outfile:
            out = {"model_name":self.model_name,
                   "event_stream":self.event_stream,
                   "current_block_index":self.current_block_index,
                   "last_context":self.context}
            outfile.write(repr(out))
            outfile.flush()
        raise SystemExit
        
    def add_block(self, block):
        block['index'] = self.current_block_index
        block['timestamp'] = time.time()
        self.event_stream.append(block)
        self.current_block_index += 1

    def add_reminder(self, reminder):
        """Reminders are trigger callbacks that get executed on each tick. They return
        a value between 0 and 1 which is compared to a threshold to determine
        if the associated reminder callback should trigger or not."""
        assert type(reminder) == dict
        assert "type" in reminder
        assert "trigger_callback" in reminder
        assert "reminder_callback" in reminder
        assert "threshold" in reminder
        assert (reminder["trigger_type"]
                in ("yes_no_logit",
                    "callback"))
        self.reminders.append(reminder)

    def remove_reminder(self, reminder):
        self.reminders.remove(reminder)

    def add_task(self, task):
        assert "type" in task
        assert "title" in task
        assert "priority" in task
        self.tasks.append(task)

    def remove_task(self, task):
        if "root" in task:
            if task["root"]:
                self.shutdown()
        self.tasks.remove(task)

    def add_observation_view(self, view):
        assert type(view) == dict
        assert "type" in view
        assert "title" in view
        assert "callback" in view
        assert type(view["callback"]) == types.FunctionType
        self.observation_views.append(view)

    def remove_observation_view(self, view):
        self.observation_views.remove(view)

    def get_block_by_index(self, index):
        return self.event_stream[index]

    def update_cache(self, key, value):
        self.cache[key] = value

    def get_cache(self, key):
        return self.cache.get(key)

    def delete_cache(self, key):
        if key in self.cache:
            del self.cache[key]

    def add_evaluation(self, evaluation):
        title, callback = evaluation["title"], evaluation["callback"]
        assert type(title) == str
        assert type(callback) == types.FunctionType
        self.current_tick.evaluations.append(evaluation)
            
    def render_context(self):
        self.context = ""
        for event_block in self.event_stream[-32:]:
            header = f'#startblock type: {event_block["type"]}\n'
            if "timestamp" in event_block:
                header += f'#timestamp {event_block["timestamp"]}\n'
            footer = '\n#endblock\n'
            if event_block["type"] in ("genesis",
                                       "bootstrap",
                                       "orientation",
                                       "action",
                                       "expectation",
                                       "evaluation"):
                self.context += (header + event_block["program"] + footer)
            elif event_block["type"] == "error":
                self.context += (header + event_block["message"] + footer)
            elif event_block["type"] == "outcome":
                self.context += (header
                                 + self.generate_outcome_table(event_block['table'])
                                 + footer)
            else:
                self.context += (header + repr(event_block) + footer)
        return self.context

    def generate_block(self, block_type, context, hint=""):
        """Generate a block and add it to the event stream."""
        prompt = f'{context}#startblock type: {block_type}\n{hint}\n'
        port = 5001
        n = 1
        n_tokens = 4096
        payload = {"n":n,
                   "temperature":1,
                   "top_k":50,
                   "repetition_penalty":1.02,
                   "max_tokens": n_tokens,
                   "model":self.model_name,
                   "prompt":prompt,
                   "stream":False,
                   "seed":random.randrange(1000000),
                   # "guided_grammar":python_grammar,
                   "stop":["\n#endblock"]}
        response = requests.post(f"http://localhost:{port}/v1/completions/",
                                 data=json.dumps(payload))
        texts = [choice["text"] for choice in response.json()["choices"]]
        if texts:
            block = {"type":block_type,
                     "program":texts[0]}
            self.add_block(block)
            try:
                compile(texts[0], f"block_{self.current_block_index}", "exec")
            except Exception as e:
                raise ValueError from e
            return block

    def add_error_block(self, error_message):
        error_block = {
            'type': 'error',
            'message': error_message
        }
        self.add_block(error_block)

    def roll_reminders(self):
        for reminder in self.reminders:
            if reminder['trigger_type'] == 'yes_no_logit':
                score = reminder['trigger_callback'](self)[0].item()
                if score >= reminder['threshold']:
                    reminder['reminder_callback'](self)
            elif reminder['trigger_type'] == 'unit_test':
                if reminder['trigger_callback'](self) >= reminder['threshold']:
                    reminder['reminder_callback'](self)

    def generate_outcome_table(self, evaluation_results):
        table = "Evaluation Results:\n"
        table += "--------------------\n"
        for program, result in evaluation_results:
            table += f"Program: {program}\n"
            table += f"Result: {result}\n"
            table += "--------------------\n"
        return table
                    
    def tick(self):
        if len(self.ticks) > 0:
            self.current_tick = Tick(self, len(self.ticks))
        # Roll reminders
        self.roll_reminders()

        # Refresh observation views
        for view in self.observation_views:
            view['callback'](self)

        # Roll for tasks to display
        active_tasks = [task for task in self.tasks if task['priority'] == 0]
        sampled_tasks = random.sample(self.tasks, min(len(self.tasks), 5))
        displayed_tasks = active_tasks + sampled_tasks

        # Format tasks into blocks
        task_blocks = [{'type': 'task-reminder', 'task': task} for task in displayed_tasks]

        # Pull the content of the observation windows into blocks
        observation_blocks = [{'type': 'observation',
                               'title': view['title'],
                               'content': view['callback'](self)} for view in self.observation_views]

        # Inject these into the event stream
        self.event_stream += (task_blocks + observation_blocks)
            
        # Render context
        self.render_context()

        # Write orientation reasoning block
        # This is your opportunity to analyze the situation based on the
        # observation, reminder, task, etc blocks. Use this moment to decide
        # what to do next.
        orientation_hint = (
            "#hint The orientation block is your opportunity to\n"
            + "# reflect on the situation, do chain of thought,\n"
            + "# summarize what has happened and what needs to\n"
            + "# be done in response, etc. It is only technically\n"
            + "# python code and does not get executed by the\n"
            + "# framework. I suggest putting your internal\n"
            + "# monologue in a triple quote block at this step."
        )
        try:
            orientation_block = self.generate_block("orientation",
                                                    self.context,
                                                    hint=orientation_hint)
        except ValueError as e:
            tb = traceback.format_exc()
            hint = ("Hint: callbacks are structured like\n\n"
                    + "def callback_name(agent):\n   "
                    + f"# code...\n   pass\nagent.add_orientation({{...}})")
            self.add_error_block(f'{hint}\n"""{tb}"""')
            return
        self.render_context()
        self.current_tick.orientation = orientation_block
        
        # Write action block
        action_hint = (
            "#hint Action blocks are where you write code to take actions.\n"
            + "# Adding and removing callbacks is a frequent kind of action you\n"
            + "# will take but it's important to remember that you can do anything\n"
            + "# a python program can do. By the way printing to standard output\n"
            + "# doesn't actually show up in the context stop doing that."
        )
        try:
            action_block = self.generate_block("action",
                                               self.context,
                                               hint=action_hint)
        except ValueError as e:
            tb = traceback.format_exc()
            hint = ("Hint: callbacks are structured like\n\n"
                    + "def callback_name(agent):\n   "
                    + f"# code...\n   pass\nagent.add_action({{...}})")
            self.add_error_block(f'{hint}\n"""{tb}"""')
            return
        self.render_context()
        self.current_tick.action = action_block

        # Write expectation block
        expectation_hint = (
            "#hint Expectation blocks are where you think about what it would\n"
            + "# look like for your action to succeed, what it would look like\n"
            + "# for it to fail. You are enumerating the expected sensory evidence\n"
            + "# that would tell you one way or another whether your action is\n"
            + "# working or not. Like the orientation this should go in triple\n"
            + "# quotes."
        )
        try:
            expectation_block = self.generate_block("expectation",
                                                    self.context,
                                                    hint=expectation_hint)
        except ValueError as e:
            tb = traceback.format_exc()
            hint = ("Hint: callbacks are structured like\n\n"
                    + "def callback_name(agent):\n   "
                    + f"# code...\n   pass\nagent.add_action({{...}})")
            self.add_error_block(f'{hint}\n"""{tb}"""')
            return
        self.render_context()            
        self.current_tick.expectation = expectation_block
        
        # Write evaluation programs
        evaluation_hint = (
            "#hint Evaluation blocks are where you write callbacks to check if\n"
            + "# your action succeeded or not based on the expectation. There are\n"
            + "# unit tests and logit evaluators. Use unit test callbacks\n"
            + "# (i.e. normal python) for symbolic manipulation tasks like\n"
            + "# checking arithmetic, the existence of a particular file, etc.\n"
            + "# Use logit evaluators for vibe-y tasks like whether a piece of\n"
            + "# writing flows well or if a source seems trustworthy. Like\n"
            + "# reminders both unit test callbacks and logit evaluators return\n"
            + "# a value between 0 and 1. Be sure to add your callback to\n"
            + "# the queue with agent.add_evaluation(title, callback)."
        )
        for _ in range(3):
            try:
                self.current_tick.evaluations.append(
                    self.generate_block("evaluation",
                                        self.context,
                                        hint=evaluation_hint)
                )
            except ValueError as e:
                tb = traceback.format_exc()
                hint = ("Hint: callbacks are structured like\n\n"
                        + "def callback_name(agent):\n   "
                        + f"# code...\n   pass\nagent.add_evaluation({{...}})")
                self.add_error_block(f'{hint}\n"""{tb}"""')
                return
            self.render_context()

        with open("confirm.txt", "w") as outfile:
            print("Confirmation waiting...")
            outfile.write(self.context)
            outfile.flush()
        while os.path.exists("confirm.txt"):
            time.sleep(1)
            
        # Execute action program
        try:
            exec(action_block['program'])
        except Exception as e:
            self.add_error_block(f"Action execution failed: {e}")
            return

        # Run evaluation programs if action did not fail
        evaluation_results = []
        for evaluation in self.current_tick.evaluations:
            try:
                result = exec(block['program'])
                evaluation_results.append((evaluation['title'], result))
            except Exception as e:
                self.add_error_block(f"Evaluation execution failed: {e}")

        # Add outcome block
        outcome_block = {
            'type': 'outcome',
            'table': evaluation_results
        }
        self.add_block(outcome_block)
        self.current_tick.outcome = outcome_block
        self.current_tick.validate()
        self.ticks.append(current_tick)


parser = ArgumentParser()
parser.add_argument("model_name", help="The model to use.")
parser.add_argument("--port", default=5000, help="The port to use for VLLM.")
parser.add_argument("--bootstrap",
                    default="bootstrap.py",
                    help="The filepath to run as bootstrap.")
args = parser.parse_args()

def simple_evaluate_outputs(score_prompt_fns, texts):
    if type(texts) == str:
        texts = [texts,]
    if type(score_prompt_fns) == types.FunctionType:
        score_prompt_fns = [score_prompt_fns,]
    scores = evaluate_outputs_vllm(args.model_name,
                                   score_prompt_fns,
                                   texts,
                                   port=args.port)
    return torch.sigmoid(scores)

agent = WeaveAgent(args.model_name)

with open("weave_agent.py") as infile:
    # Genesis block
    genesis_block = {
        'type': 'genesis',
        'program': infile.read()
    }
    agent.add_block(genesis_block)

with open(args.bootstrap) as infile:
    # Bootstrap block
    bootstrap_block = {
        'type': 'bootstrap',
        'program': infile.read()
    }
    agent.add_block(bootstrap_block)
    exec(bootstrap_block["program"])

# Run the agent
while True:
    agent.tick()
    time.sleep(1)  # Simulate tick interval
