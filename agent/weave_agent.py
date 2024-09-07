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
from typing import List, Dict, Optional, Any
from functools import partial
from tqdm import tqdm
from rich import print as rprint
from weave import generate_outputs_vllm, evaluate_outputs_vllm
from weave import bayesian_evaluate_outputs_vllm
from weave import make_score_prompt_vllm, make_bayes_score_prompt_vllm
from weave import weave_tree_search, TreeNode

def make_simple_bayes_score_prompt(question: str):
    """Simplify the process of making a bayesian weave evaluator question prompt
    maker so that it's just a matter of passing a question for the weave-agent."""
    template = ("{response}\n\n"
                + "# Answer yes or no and only yes or no to the following.\n"
                + "# question about the incomplete code block above.\n"
                + "# Keep in mind the following question is being asked as part\n"
                + "# of a Monte Carlo Tree Search so the above is usually a work in progress.\n"
                + "# You're really being asked something like *will this trajectory*\n"
                + "# eventually have quality X or satisfy predicate Y?\n"
                + f"#q: {{parent_q}}\n# {question}")
    return partial(make_bayes_score_prompt_vllm, template, "", "")
    
def make_simple_score_prompt(question: str):
    """Simplify the process of making a weave evaluator question prompt maker so
    that it's just a matter of passing a question for the weave-agent."""
    template = ("{response}\n\n"
                + "# Answer yes or no and only yes or no to the following.\n"
                + "# question about the incomplete code block above.\n"
                + "# Keep in mind the following question is being asked as part\n"
                + "# of a Monte Carlo Tree Search so the above is usually a work in progress.\n"
                + "# You're really being asked something like *will this trajectory*\n"
                + "# eventually have quality X or satisfy predicate Y?\n"
                + f"#q: {question}")
    return partial(make_score_prompt_vllm, template, "", "")


class WeaveKanbanTask:
    STATUSES = ['idle', 'going', 'completed', 'blocked', 'aborted']
    ABBREVIATIONS = {'idle': 'I', 'going': 'G', 'completed': 'C', 'blocked': 'B', 'aborted': 'A'}

    def __init__(self, task_id: int, title: str,
                 description: str = "", status: str = "idle",
                 blocked_on: Optional[List[str]] = None):
        self.id = task_id
        self.title = title
        self.description = description
        if status not in WeaveKanbanTask.STATUSES:
            raise ValueError(f'Status "{status}" not valid.')
        self.status = status
        self.history: List[Dict[str, str]] = [{'status': self.status,
                                               'explanation': 'Task created'}]
        self.blocked_on: List[str] = blocked_on

    def change_status(self, new_status: str, explanation: str,
                      blocked_on: Optional[List[str]] = None) -> None:
        if new_status == self.status:
            return

        if new_status not in self.STATUSES:
            raise ValueError(f"Invalid status: {new_status}")

        if new_status == 'blocked' and not blocked_on:
            raise ValueError("Blocked status requires a list of tasks it's blocked on")

        self.status = new_status
        self.history.append({'status': new_status, 'explanation': explanation})

        if new_status == 'blocked':
            self.blocked_on = blocked_on
        else:
            self.blocked_on = []

    def idle(self, explanation: str) -> None:
        self.change_status('idle', explanation)

    def going(self, explanation: str) -> None:
        self.change_status('going', explanation)

    def completed(self, explanation: str) -> None:
        self.change_status('completed', explanation)

    def blocked(self, explanation: str, blocked_on: List[str]) -> None:
        self.change_status('blocked', explanation, blocked_on)

    def aborted(self, explanation: str) -> None:
        self.change_status('aborted', explanation)

    def view_task(self) -> str:
        history = "\n".join([f"- {h['status']}: {h['explanation']}" for h in self.history])
        return f"ID: {self.id}\nTitle: {self.title}\nDescription: {self.description}\nMetadata: {self.blocked_on}\nHistory:\n{history}"

    def abbreviated_history(self) -> str:
        letter_history = ' '.join([self.ABBREVIATIONS[h['status']] for h in self.history])
        # Display full name of final status to help LLM read it past tokenizer
        return letter_history[:-1] + self.history[-1]['status'].title()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'metadata': self.metadata,
            'status': self.status,
            'history': self.history,
            'blocked_on': self.blocked_on
        }

    @classmethod
    def from_dict(cls, task_dict: Dict[str, Any]) -> 'WeaveKanbanTask':
        task = cls(
            task_id=task_dict['id'],
            title=task_dict['title'],
            description=task_dict['description'],
            metadata=task_dict['metadata']
        )
        task.status = task_dict['status']
        task.history = task_dict['history']
        task.blocked_on = task_dict['blocked_on']
        return task

class WeaveKanban:
    def __init__(self):
        self.tasks: List[WeaveKanbanTask] = []
        self.next_id = 1

    def add_task(self, title: str, description: str = "", status: str = "idle",
                 blocked_on: Optional[List[str]] = None) -> None:
        task = WeaveKanbanTask(self.next_id, title, description, status, blocked_on)
        self.tasks.append(task)
        self.next_id += 1

    def get_task(self, task_id: int) -> Optional[WeaveKanbanTask]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def view_board(self) -> str:
        table = [[task.id, task.title, task.abbreviated_history()]
                 for task in self.tasks if task.status not in ["completed", "aborted"]]
        headers = ['ID', 'Title', 'History']
        col_widths = [max(len(str(item)) for item in col) for col in zip(*table, headers)]

        def format_row(row: List[Any]) -> str:
            return ' | '.join(f"{item:<{col_widths[i]}}" for i, item in enumerate(row))

        header_row = format_row(headers)
        separator_row = ' | '.join('-' * width for width in col_widths)
        table_rows = '\n'.join(format_row(row) for row in table)

        return f"{header_row}\n{separator_row}\n{table_rows}"

    def to_json(self) -> str:
        return json.dumps([task.to_dict() for task in self.tasks], indent=2)

    def from_json(self, json_str: str) -> None:
        task_dicts = json.loads(json_str)
        self.tasks = [WeaveKanbanTask.from_dict(task_dict) for task_dict in task_dicts]
        self.next_id = max([task.id for task in self.tasks], default=0) + 1
    

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
        elif "program" not in self.action_setup:
            raise TypeError("Tick action has no program.")
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
        self.pinned_events = [0,1]
        self.current_tick = Tick(self, 0)
        self.ticks = []
        self.current_block_index = 0
        self.reminders = []
        self.tasks = WeaveKanban()
        self.current_task = None
        self.observation_views = []
        self.cache = {}
        self.context = ""

    def shutdown(self):
        """The agent shutdown routine. This should be called when the agent's 
        root task has been resolved, the root task is deemed intractable, or the
        agent has wandered off so far it can't find its way back to the task."""
        if not os.path.exists("/app/weave-agent-logs"):
            os.mkdir("/app/weave-agent-logs")
        with open(f"/app/weave-agent-logs/{round(time.time())}/log.json", "w") as outfile:
            out = {"model_name":self.model_name,
                   "event_stream":self.event_stream,
                   "current_block_index":self.current_block_index,
                   "last_context":self.context}
            json.dump(out, outfile)
            outfile.flush()
        # Temp code to stop docker from shutting down
        while 1:
            time.sleep(30)
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

    def add_task(self, title, description, status, blocked_on=None):
        self.tasks.add_task(title,
                            description,
                            status,
                            blocked_on)

    def add_action(self, title, callback):
        assert type(title) == str
        assert type(callback) == types.FunctionType
        self.current_tick.action = {"type":"action",
                                    "title":title,
                                    "callback":callback}

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

    def add_evaluation(self, title, callback):
        assert type(title) == str
        assert type(callback) == types.FunctionType
        self.current_tick.evaluations.append({"type":"evaluation",
                                              "title":title,
                                              "callback":callback})
            
    def render_context(self):
        self.context = ""
        context_blocks = []
        history_len = 30
        for index in self.pinned_events:
            if (len(self.event_stream) - index) > history_len:
                context_blocks.append(self.event_stream[index])
        context_blocks += self.event_stream[-history_len:]
        for event_block in context_blocks:
            header = f'#startblock type: {event_block["type"]}\n'
            if "timestamp" in event_block:
                header += f'#timestamp {event_block["timestamp"]}\n'
            footer = ""
            if "q" in event_block:
                yes_p = torch.sigmoid(torch.tensor(event_block["score"])).item()
                no_p = 1 - yes_p
                yes_p, no_p = round(yes_p, 5), round(no_p, 5)
                answer = random.choices(["Yes.", "No."], weights=[yes_p, no_p])[0]
                if answer == "Yes.":
                    prob = f"({round(yes_p * 100, 5)}%)"
                else:
                    prob = f"({round(no_p * 100, 5)}%)"
                # TODO: Turn this back on. Goodharts too much right now. 
                # footer += f'\n#q: {event_block["q"]} {answer} {prob}'
            footer += '\n#endblock\n'
            if event_block["type"] in ("genesis",
                                       "action",
                                       "expectation",
                                       "observation_inference",
                                       "evaluation"):
                self.context += (header + "\n" + event_block["program"] + footer)
            elif event_block["type"] == "bootstrap":
                footer += "# END OF DEMO. Starting on the next tick you have\n"
                footer += "# full control. Wake up.\n"
                self.context += (header + "\n" + event_block["program"] + footer)
            elif event_block["type"] == "orientation":
                header += f"# Current Working Directory: {os.getcwd()}\n"
                self.context += (header + "\n" + event_block["program"] + footer)
            elif event_block["type"] == "task_inference":
                task_title = f"({self.current_task.id}) {self.current_task.title}"
                task_status = f"({self.current_task.status}) "
                task_status += f"{self.current_task.history[-1]['explanation']}"
                header += f"# Current Task: {task_title}\n"
                header += f"# Task Status: {task_status}\n"
                self.context += (header + "\n" + event_block["program"] + footer)
            elif event_block["type"] == "task-reminder":
                self.context += (header + "\n" + event_block["task"] + footer)
            elif event_block["type"] == "error":
                header += "# WARNING: ERROR MEANS TICK DID NOT FINISH EXECUTION\n"
                header += "# ADDRESS ERROR IN NEXT TICK BEFORE PROCEEDING\n"
                self.context += (header + "\n" + event_block["message"] + footer)
            elif event_block["type"] == "outcome":
                footer += f"# Starting tick #{len(self.ticks) + 1} "
                footer += f"with block #{self.current_block_index}\n"
                self.context += (header + "\n" 
                                 + self.generate_outcome_table(event_block['table'])
                                 + footer)
            else:
                self.context += (header + repr(event_block) + footer)
        return self.context

    def generate_block(self, block_type, context, eval_questions, weave_params, hint=""):
        """Generate a block and add it to the event stream."""
        prompt = f'<s> [INST]{context}[/INST]#startblock type: {block_type}\n{hint}\n\n'
        # Narrow incidence of structurally wrong blocks by premising correct prefix
        if block_type in {"orientation", "expectation",
                          "task_inference", "observation_inference"}:
            prefix = '"""'
        elif block_type in {"action", "evaluation"}:
            prefix = "def "
        else:
            prefix = ""
        prompt += prefix
        port = 5001
        stopstrings = ["\n#q: ", "\n# q:", "#endblock", "#startblock"]
        generate_fn = partial(generate_outputs_vllm,
                              self.model_name,
                              port=port,
                              stop=stopstrings)
        score_prompt_fns = []
        # TODO: Use the full set of questions somehow?
        score_prompt_fns.append(make_simple_score_prompt(eval_questions[0]))
        evaluate_fn = partial(evaluate_outputs_vllm,
                              self.model_name,
                              score_prompt_fns,
                              port=port)
        
        
        tree = TreeNode(prompt)
        wp = weave_params
        # First try a simple rejection sampling
        branches = weave_tree_search(tree=tree,
                                     generate_fn=partial(generate_fn,
                                                         n_tokens=768),
                                     evaluate_fn=evaluate_fn,
                                     budget=32,
                                     round_budget=32,
                                     n_expand=32,
                                     beam_width=1,
                                     max_lookahead=1,
                                     temperature=0.01)
        do_long = False
        if (branches[-1].score < 3.0):
            do_long = True
        try:
            program = branches[-1].branch_text()
            program = prefix + program.strip()
            compile(program, f"block_{self.current_block_index}", "exec")
        except Exception as e:
            do_long = True
        # If rejection sampling fails do full search
        if do_long:
            tree = TreeNode(prompt)
            branches = weave_tree_search(tree=tree,
                                         generate_fn=partial(generate_fn,
                                                             n_tokens=wp["weave_n_tokens"]),
                                         evaluate_fn=evaluate_fn,
                                         budget=wp["weave_budget"],
                                         round_budget=wp["weave_round_budget"],
                                         n_expand=wp["weave_n_expand"],
                                         beam_width=wp["weave_beam_width"],
                                         max_lookahead=wp["weave_max_lookahead"],
                                         temperature=wp["weave_temperature"]) 
            program = branches[-1].branch_text()
            # Check we finished writing the code block and extract first block
            stop_indices = []
            for stopstring in stopstrings:
                candidate = program.find(stopstring)
                if candidate > -1:
                    stop_indices.append(candidate)
            if stop_indices:
                stop_index = min(stop_indices)
                program = prefix + program[:stop_index].strip()
            else:
                program = prefix + program.strip()
        block = {"type":block_type,
                 "program":program,
                 "q":eval_questions[0],
                 "score":branches[-1].score.item()}
        self.add_block(block)
        try:
            compile(program, f"block_{self.current_block_index}", "exec")
        except Exception as e:
            raise ValueError from e
        rprint(f"Finished writing block #[cyan]{self.current_block_index}[/cyan] of type [cyan]{block_type}[/cyan]")
        print(block["program"])
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
        self.current_tick = Tick(self, len(self.ticks))
        # Roll reminders
        self.roll_reminders()

        observations = []
        # Refresh observation views
        for view in self.observation_views:
            try:
                observations.append(view['callback'](self))
            except Exception as e:
                tb = traceback.format_exc()
                self.add_error_block(
                    f"Observation callback '{view['title']}' failed: {tb}"
                )
                
        task_reminder_body = ""
        
        if self.current_task:
            task_reminder_body += "# Current Task:\n"
            task_reminder_body += ('"""\n' + self.current_task.view_task() + '\n"""\n')
        task_reminder_body += "# Kanban Board:\n"
        task_reminder_body += ('"""\n' + self.tasks.view_board() + '\n"""')

        # Format tasks into blocks
        task_blocks = [{'type': 'task-reminder', 'task': task_reminder_body},]

        # Pull the content of the observation windows into blocks
        observation_blocks = [{'type': 'observation',
                               'title': view['title'],
                               'content': observation} for observation in observations]

        # Inject these into the event stream
        self.event_stream += (task_blocks + observation_blocks)
            
        # Render context
        self.render_context()


        def do_tick_block(self, block_type, hint, wp_update):
            weave_params = {"weave_n_tokens":256, "weave_budget":72,
                            "weave_round_budget":24, "weave_n_expand":16,
                            "weave_beam_width":1, "weave_max_lookahead":3,
                            "weave_temperature":0.2}
            weave_params.update(wp_update)
            with open(f"/app/eval_rubrics/{block_type}.txt") as infile:
                inference_questions = infile.read().strip().splitlines()
            try:
                block = self.generate_block(block_type,
                                            self.context,
                                            inference_questions,
                                            weave_params,
                                            hint=hint)
            except ValueError as e:
                tb = traceback.format_exc()
                hint = ("Hint: callbacks are structured like\n\n"
                        + "def callback_name(agent):\n   "
                        + f"# code...\n   pass\nagent.add_orientation({{...}})")
                self.add_error_block(f'{hint}\n"""{tb}"""')
                return
            self.render_context()
            return block
            
        # Write orientation reasoning block
        # This is your opportunity to analyze the situation based on the
        # observation, reminder, task, etc blocks. Use this moment to decide
        # what to do next.
        orientation_hint = (
            "#hint The orientation block is my opportunity to\n"
            + "# reflect on the situation, do chain of thought,\n"
            + "# summarize what has happened and what needs to\n"
            + "# be done in response, etc. It is only technically\n"
            + "# python code and does not get executed by the\n"
            + "# framework. I suggest putting my internal\n"
            + "# monologue in a triple quote block at this step."
        )
        mcts_params = {"weave_n_tokens":256, "weave_budget":288,
                       "weave_round_budget":96, "weave_n_expand":32}
        orientation_block = do_tick_block(agent,
                                          "orientation",
                                          orientation_hint,
                                          mcts_params)
        if orientation_block:
            self.current_tick.orientation = orientation_block
        else:
            return

        # Task inference block
        task_inference_hint = (
            "# In the task inference stage I change the status of tasks on the kanban\n"
            + "# board, add new tasks if necessary, etc. It's important to keep my kanban\n"
            + "# up to date so that I'm presented with the correct task state at the\n"
            + "# start of each tick. In particular I should:\n"
            + "# 1) Write python to update the kanban board.\n"
            + "# 2) Set the current task if it needs to be changed or I've completed\n"
            + "# it.\n"
            + "# 3) Change the status of any tasks that have been completed, made irrelevant,\n"
            + "# etc.\n" 
            + "# 4) If all tasks are completed, shutdown.\n"
            + "# Keep in mind that the kanban will not actually be updated until the next tick."
        )
        task_inference_block = do_tick_block(agent,
                                             "task_inference",
                                             task_inference_hint,
                                             {})
        if task_inference_block:
            self.current_tick.task_inference = task_inference_block
        else:
            return
        
        # Write action block
        action_hint = (
            "#hint Action blocks are where I write code to take actions.\n"
            + "# Write a callback to further my goal(s) based on the orientation\n"
            + "# block and set up the callback to be executed with the agent.add_action()\n"
            + "# method. I must write a callback and then set it up to be executed\n"
            + "# later with agent.add_action() or the tick will not be accepted.\n"
            + "# It's important to remember that my callback can do anything\n"
            + "# a python program can do through side effects in the external\n" 
            + "# computable environment. If I need to import a new module make sure\n"
            + "# to do it inside the callback because the tick gets executed in a\n"
            + "# local context."
        )
        action_block = do_tick_block(agent,
                                     "action",
                                     action_hint,
                                     {})
        if action_block:
            self.current_tick.action_setup = action_block
        else:
            return
            
        # Write expectation block
        expectation_hint = (
            "#hint Expectation blocks are where I think about what it would\n"
            + "# look like for my action to succeed, what it would look like\n"
            + "# for it to fail. I am enumerating the expected sensory evidence\n"
            + "# that would tell me one way or another whether my action is\n"
            + "# working or not. Like the orientation this should go in triple\n"
            + "# quotes."
        )
        expectation_block = do_tick_block(agent,
                                          "expectation",
                                          expectation_hint,
                                          {})
        if expectation_block:
            self.current_tick.expectation = expectation_block
        else:
            return
            
        # Observation Inference Block
        observation_inference_hint = (
            "# In the observation inference stage I manage the observation\n"
            + "# callbacks that fetch information on each tick. Since I just\n"
            + "# formulated my expectations now is my opportunity to review\n"
            + "# and change the observation blocks that will be presented on the\n"
            + "# next tick. Remove callbacks that are no longer necessary,\n"
            + "# prepare callbacks that will be useful to help me render judgment\n"
            + "# on whether the action succeeded on the next tick."
        )
        observation_inference_block = do_tick_block(agent,
                                                    "observation_inference",
                                                    observation_inference_hint,
                                                    {})
        if observation_inference_block:
            self.current_tick.observation_inference = observation_inference_block
        else:
            return
            
        # Write evaluation programs
        evaluation_blocks = []
        evaluation_hint = (
            "#hint Evaluation blocks are where I write callbacks to check if\n"
            + "# my action succeeded or not based on the expectation. There are\n"
            + "# unit tests and logit evaluators. Use unit test callbacks\n"
            + "# (i.e. normal python) for symbolic manipulation tasks like\n"
            + "# checking arithmetic, the existence of a particular file, etc.\n"
            + "# Use logit evaluators for vibe-y tasks like whether a piece of\n"
            + "# writing flows well or if a source seems trustworthy. Like\n"
            + "# reminders both unit test callbacks and logit evaluators return\n"
            + "# a value between 0 and 1. I should be sure to add my callback to\n"
            + "# the queue with agent.add_evaluation(title, callback)."
        )
        for _ in range(1):
            eval_block = do_tick_block(agent,
                                       "evaluation",
                                       evaluation_hint,
                                       {})
            if eval_block:
                evaluation_blocks.append(eval_block)
            else:
                return
            
        # Execute task updates
        try:
            exec(task_inference_block['program'])
        except Exception as e:
            self.add_error_block(f"# Task updates failed: {e}")
            return
            
        # Set up action callback
        try:
            exec(action_block['program'])
        except Exception as e:
            self.add_error_block(f"# Action execution failed: {e}")
            return

        # Set up evaluation callbacks
        for evaluation_block in evaluation_blocks:
            try:
                exec(evaluation_block['program'])       
            except Exception as e:
                self.add_error_block("# Evaluation setup execution failed: \n"
                                     + f'"""{e}"""')
                return
        self.current_tick.evaluation_setup = evaluation_blocks

        # Run action callback
        try:
            action_result = self.current_tick.action["callback"](self)
        except Exception as e:
            action_result = traceback.format_exc()
        
        # Run evaluation callbacks
        evaluation_results = []
        for evaluation in self.current_tick.evaluations:
            try:
                result = evaluation["callback"](self)
                evaluation_results.append((evaluation['title'], result))
            except Exception as e:
                tb = traceback.format_exc()
                evaluation_results.append((evaluation['title'], "ERROR"))
                self.add_error_block("# Evaluation failed: \n"
                                     + f'"""{tb}"""')

        outcomes =  []
        try:
            outcomes += [(self.current_tick.action["title"],action_result),]
        except AttributeError:
            outcomes += [("[No action specified with agent.add_action()]", "ERROR"),]
        outcomes += evaluation_results
                
        # Add outcome block
        outcome_block = {
            'type': 'outcome',
            'table': outcomes
        }
        self.add_block(outcome_block)
        self.current_tick.outcome = outcome_block
        try:
            self.current_tick.validate()
        except Exception as e:
            tb = traceback.format_exc()
            self.add_error_block("# Tick validation failed: \n"
                                 + f'"""{tb}"""')
            self.current_tick.valid = False
        self.ticks.append(self.current_tick)
        if len(self.ticks) % 2 == 0:
            with open(f"/app/event_trace_{round(time.time())}.json", "w") as outfile:
                json.dump(self.event_stream, outfile)


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

def simple_bayes_evaluate_outputs(parent_q, questions, texts):
    if type(texts) == str:
        texts = [texts,]
    score_prompt_fns = [make_simple_bayes_score_prompt(question)
                        for question in questions]
    scores = bayesian_evaluate_outputs_vllm(args.model_name,
                                            parent_q,
                                            score_prompt_fns,
                                            texts,
                                            port=args.port)
    return scores

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
