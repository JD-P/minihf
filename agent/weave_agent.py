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
import ast
import types
import asyncio
import traceback
import requests
import torch
from argparse import ArgumentParser
from typing import List, Dict, Optional, Any
from functools import partial
from tqdm import tqdm
from rich import print as rprint
from transformers import AutoTokenizer
import tantivy
from tantivy import Index, SchemaBuilder
from weave import generate_outputs_vllm, evaluate_outputs_vllm
from weave import bayesian_evaluate_outputs_vllm
from weave import make_score_prompt_vllm, make_bayes_score_prompt_vllm
from weave import weave_tree_search, TreeNode
from render_block import render_block
from block_generators import generate_block_inner
from block_generators import make_simple_bayes_score_prompt, make_simple_score_prompt


class WeaveKanbanTask:
    STATUSES = ['idle', 'going', 'completed', 'blocked', 'aborted']
    ABBREVIATIONS = {'idle': 'I', 'going': 'G', 'completed': 'C', 'blocked': 'B', 'aborted': 'A'}

    def __init__(self, kanban, task_id: int, title: str,
                 description: str = "", status: str = "idle",
                 blocked_on: Optional[List[str]] = None):
        self.kanban = kanban
        self.id = int(task_id)
        self.title = str(title)
        self.description = description
        # Set initial status
        self.history: List[Dict[str, str]] = []
        try:
            if status == 'blocked':
                getattr(self, status)('Task created', blocked_on)
            else:
                getattr(self, status)('Task created')
                self.blocked_on: List[int] = blocked_on
        except:
            raise ValueError(f'Status "{status}" not valid.')
        self.evaluations = []

    def change_status(self, new_status: str, explanation: str,
                      blocked_on: Optional[List[int]] = None) -> None:
        try:
            if new_status == self.status:
                return
        except AttributeError:
            pass

        if new_status not in self.STATUSES:
            raise ValueError(f"Invalid status: {new_status}")

        if new_status == 'blocked' and not blocked_on:
            raise ValueError("Blocked status requires a list of tasks it's blocked on")
        if new_status == 'blocked':
            for task_id in blocked_on:
                try:
                    assert self.kanban.get_task(task_id)
                except:
                    raise ValueError(f"Tried to block on nonexistent task {task_id}!")
        self.status = new_status
        self.history.append({'status': new_status, 'explanation': explanation})

        if new_status == 'blocked':
            self.blocked_on = blocked_on
        else:
            self.blocked_on = []

    def add_evaluation(self, title, callback):
        assert type(title) == str
        assert type(callback) == types.FunctionType
        self.evaluations.append({"type":"evaluation",
                                 "title":title,
                                 "callback":callback})
        
    def idle(self, explanation: str) -> None:
        self.change_status('idle', explanation)

    def going(self, explanation: str) -> None:
        self.change_status('going', explanation)

    def run_evaluations(self):
        results = {}
        for evaluation in self.evaluations:
            try:
                result = evaluation["callback"](self.kanban.agent)
            except Exception as e:
                result = traceback.format_exc()
            results[evaluation["callback"].__name__] = result
        return results
        
    def completed(self, explanation: str) -> None:
        # Run evaluation callbacks
        evaluation_results = self.run_evaluations()
        for evaluation in self.evaluations:
            try:
                assert evaluation_results[evaluation["callback"].__name__] == True
            except Exception as e:
                msg = (f"# Unable To .completed() Task '{self.title}' Due To Failed Test: \n"
                       + "# If you're seeing this it's because you tried to do\n"
                       + "# .completed() on a task and its test suite failed.\n"
                       + f"# The failing test is '{evaluation['title']}'\n")
                tb = traceback.format_exc()
                self.kanban.agent.failure_stage = "'{self.title}' .completed() test suite"
                raise ValueError(msg + f'"""{tb}"""')
        if self.kanban.agent.debugging:
            raise ValueError("Can't complete a task while error in last tick.")
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
            'status': self.status,
            'history': self.history,
            'blocked_on': self.blocked_on
        }

    @classmethod
    def from_dict(cls, kanban, task_dict: Dict[str, Any]) -> 'WeaveKanbanTask':
        task = cls(
            kanban,
            task_id=task_dict['id'],
            title=task_dict['title'],
            description=task_dict['description'],
        )
        task.status = task_dict['status']
        task.history = task_dict['history']
        task.blocked_on = task_dict['blocked_on']
        return task

class WeaveKanban:
    def __init__(self, agent):
        self.agent = agent
        self.tasks: List[WeaveKanbanTask] = []
        self.next_id = 1

    def add_task(self, title: str, description: str = "", status: str = "idle",
                 blocked_on: Optional[List[str]] = None) -> None:
        task = WeaveKanbanTask(self, self.next_id, title, description, status, blocked_on)
        self.tasks.append(task)
        self.next_id += 1

    def get_task(self, task_id: int) -> Optional[WeaveKanbanTask]:
        task_id = int(task_id)
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
        table_rows = ""
        for row in table:
            table_rows += format_row(row) + "\n"
            if self.get_task(row[0]) == self.agent.current_task:
                evaluation_results = self.get_task(row[0]).run_evaluations()
                for evaluation in self.get_task(row[0]).evaluations:
                    evaluation_fn = evaluation["callback"]
                    result = evaluation_results[evaluation_fn.__name__]
                    table_rows += (" " * 7) + " - " + evaluation_fn.__name__ + ": " + str(result) + "\n"
        return f"{header_row}\n{separator_row}\n{table_rows}"

    def unblock(self) -> None:
        """Automatically unblock tasks when their blockers are completed."""
        for task in self.tasks:
            if task.status == "blocked":
                if False not in [self.get_task(task_id).status == "completed"
                                 for task_id in task.blocked_on]:
                    task.idle("Automatically unblocked by completing blockers")
                    

    def to_json(self) -> str:
        return json.dumps([task.to_dict() for task in self.tasks], indent=2)

    def from_json(self, json_str: str) -> None:
        task_dicts = json.loads(json_str)
        self.tasks = [WeaveKanbanTask.from_dict(self, task_dict) for task_dict in task_dicts]
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
        elif "body" not in self.action_setup:
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
        # Pin genesis and bootstrap so agent knows how to use framework
        self.pinned_events = [0, 1]
        self.current_tick = Tick(self, 0)
        self.ticks = []
        self.current_block_index = 0
        self.reminders = []
        self.debugging = False
        self.failure_stage = "event stream"
        self.tasks = WeaveKanban(self)
        self.current_task = None
        self.observation_views = []
        self.tools = {}
        self.cache = {}
        self.context = ""

        schema_builder = SchemaBuilder()
        schema_builder.add_text_field("type", stored=True)
        schema_builder.add_text_field("render", stored=True)
        schema_builder.add_text_field("q", stored=True)
        schema_builder.add_float_field("score", stored=True)
        schema_builder.add_integer_field("index", stored=True)
        schema_builder.add_float_field("timestamp", stored=True)
        schema_builder.add_text_field("tags", stored=True)

        self.bm25_schema = schema_builder.build()
        
        if not os.path.exists("memories"):
            os.mkdir("memories")
        if not os.path.exists("memories/bm25"):
            os.mkdir("memories/bm25")
        self.bm25_index = Index(self.bm25_schema, path="./memories/bm25")

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
        if block['type'] == 'orientation':
            block['metadata'] = {
                "tick_number":len(self.ticks) + 1,
                "block_index":self.current_block_index,
                "working_directory":os.getcwd()
            }
        if block['type'] == 'task-inference':
            try:
                block['metadata'] = {
                    "task_id":self.current_task.id,
                    "task_title":self.current_task.title,
                    "task_status":self.current_task.status,
                    "task_explanation":self.current_task.history[-1]['explanation']
                }
            except AttributeError:
                explanation = ("Right now there is no task selected. You can "
                               + "select a task with agent.current_task = "
                               + "agent.tasks.get_task(task_id)")
                block['metadata'] = {
                    "task_id":-1,
                    "task_title": "No Task Set As Current Task",
                    "task_status": "nonexistent",
                    "task_explanation": explanation
                }
        if "q" not in block:
            block["q"] = ""
        if "score" not in block:
            #TODO: Make actual score function for observations, task reminders etc
            block["score"] = 2
        if "tags" not in block:
            #TODO: Make actual tagging function
            block["tags"] = ["placeholder",]
        self.event_stream.append(block)

        if block["type"] not in {"genesis", "bootstrap"}:
            writer = self.bm25_index.writer()
            writer.add_document(tantivy.Document(
                type=block["type"],
                render=render_block(block),
                q=block["q"],
                score=block["score"],
                index=block["index"],
                timestamp=block["timestamp"],
                tags=" ".join(block["tags"]),
            ))
            writer.commit()
        
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

    def add_observation_view(self, title, callback):
        view = {"type":"observation",
                "title":title,
                "callback":callback}
        assert type(callback) in [types.FunctionType, types.MethodType]
        self.observation_views.append(view)

    def remove_observation_view(self, view_title):
        views = [view for view in self.observation_views if view['title'] == view_title]
        for view in views:
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
            self.context += render_block(event_block)
        return self.context

    def generate_block(self, block_type, context, eval_questions, weave_params, hint=""):
        """Generate a block and add it to the event stream."""
        return generate_block_inner(self, block_type, context, eval_questions, weave_params, hint)

    def add_error_block(self, error_message):
        self.debugging = True
        error_block = {
            'type': 'error',
            'message': error_message
        }
        self.add_block(error_block)
                    
    def tick(self):
        self.tasks.unblock()
        try:
            if "ERROR" in [outcome[1] for outcome in
                           self.current_tick.outcome["table"]]:
                self.debugging = True
        except AttributeError:
            self.debugging = True
        self.current_tick = Tick(self, len(self.ticks))

        observations = []
        # Refresh observation views
        for view in self.observation_views:
            try:
                observations.append((view['title'], view['callback'](self)))
            except Exception as e:
                tb = traceback.format_exc()
                self.add_error_block(
                    f"# Observation callback '{view['title']}' failed:\n"
                    + f'"""{tb}"""'
                )
                
        task_reminder_body = ""

        try:
            if self.current_task:
                task_reminder_body += "# Current Task:\n"
                task_reminder_body += ('"""\n' + self.current_task.view_task() + '\n"""\n')
            task_reminder_body += "# Kanban Board:\n"
            task_reminder_body += ('"""\n' + self.tasks.view_board() + '\n"""')
        except Exception as e:
            tb = traceback.format_exc()
            self.failure_stage = "task reminder"
            self.add_error_block(
                f"# TASK REMINDERS OFFLINE DUE TO CORRUPTED DATA. DID YOU DIRECTLY\n"
                + "# MODIFY TASK ATTRIBUTES? YOU MUST RESOLVE THIS IMMEDIATELY OR\n"
                + "# YOU WILL LOSE TRACK OF WHAT YOU'RE DOING. INVESTIGATE agent.tasks\n"
                + "# AND ATTRIBUTES ON TASKS INSIDE."
                + f'"""{tb}"""'
            )
            
        # Format tasks into blocks
        task_blocks = [{'type': 'task-reminder', 'body': task_reminder_body},]

        # Pull the content of the observation windows into blocks
        observation_blocks = [{'type': 'observation',
                               'title': observation[0],
                               'body': observation[1]} for observation in observations]

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
            rprint(f"Writing block #[cyan]{self.current_block_index}[/cyan] of type [cyan]{block_type}[/cyan]")
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
                self.failure_stage = block_type
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
            "#hint Keep this simple. In the task inference stage I change the status\n"
            + "# of tasks on the kanban board. If the current task doesn't need\n"
            + "# changed I set it to .going() with a short summary of where the\n" 
            + "# task is at like .going('Fixing HTTP errors...') If the current task\n"
            + "# is complete I do .completed() to prompt the kanban to check that it\n"
            + "# really is complete and set a new task in the statement after it by\n"
            + "# looking back at the kanban board and doing\n"
            + "# agent.tasks.get_task(next_task_id) where next_task_id is a integer\n"
            + "# literal pointing at the right next task."
        )
        task_inference_block = do_tick_block(agent,
                                             "task-inference",
                                             task_inference_hint,
                                             {})
        if task_inference_block:
            self.current_tick.task_inference = task_inference_block
        else:
            return

        # Execute task updates
        try:
            exec(task_inference_block['body'])
        except Exception as e:
            tb = traceback.format_exc()
            self.add_error_block(f"# task-inference failed:\n"
                                 + f'"""{tb}"""')
            self.failure_stage = "task-inference"
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
        for i in range(3):
            action_block = do_tick_block(agent,
                                         "action",
                                         action_hint,
                                         {})
            if action_block:
                self.current_tick.action_setup = action_block
            else:
                # TODO: Dynamic hints by having the model or external entities
                # such as user analyze the situation and suggest a course of action
                action_hint = ("#hint Rewrite the block keeping the above error in mind.\n"
                               + f"# {3 - (i+1)} attempts remaining.")
                continue

            # Set up action callback
            try:
                exec(action_block['body'])
                failed = False
            except Exception as e:
                tb = traceback.format_exc()
                self.add_error_block("# Action execution failed:\n"
                                     + f'"""{tb}"""')
                self.failure_stage = "action"
                action_hint = ("#hint Rewrite the block keeping the above error in mind.\n"
                               + f"# {3 - (i+1)} attempts remaining.")
                failed = True
                continue
            break
                
        if not hasattr(self.current_tick, "action_setup") or failed:
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
                                                    "observation-inference",
                                                    observation_inference_hint,
                                                    {})
        if observation_inference_block:
            self.current_tick.observation_inference = observation_inference_block
        else:
            return

        # Execute observation updates
        try:
            exec(observation_inference_block['body'])
        except Exception as e:
            tb = traceback.format_exc()
            self.add_error_block("# observation-inference failed:\n"
                                 + f'"""{tb}"""')
            self.failure_stage = "observation-inference"
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
        # TODO: Make this multiple blocks again
        for _ in range(1):
            for i in range(3):
                eval_block = do_tick_block(agent,
                                           "evaluation",
                                           evaluation_hint,
                                           {})
                if eval_block:
                    evaluation_blocks.append(eval_block)
                else:
                    # TODO: Dynamic hints by having the model or external entities
                    # such as user analyze the situation and suggest a course of action
                    evaluation_hint = ("#hint Rewrite the block keeping the above error in mind.\n"
                                       + f"# {3 - (i+1)} attempts remaining.")
                    continue

                # Set up evaluation callbacks
                for evaluation_block in evaluation_blocks:
                    try:
                        exec(evaluation_block['body'])
                        failed = False
                    except Exception as e:
                        tb = traceback.format_exc()
                        self.add_error_block("# Evaluation setup execution failed:\n"
                                             + f'"""{tb}"""')
                        self.failure_stage = "evaluation"
                        evaluation_hint = ("#hint Rewrite the block keeping the above error in mind.\n"
                                           + f"# {3 - (i+1)} attempts remaining.")
                        failed = True
                        continue
                break
        if not evaluation_blocks or failed:
            return
        else:
            self.current_tick.evaluation_setup = evaluation_blocks

        # TODO: Figure out how I want to allow retries on this phase
        # Run action callback
        try:
            action_result = self.current_tick.action["callback"](self)
        except Exception as e:
            action_result = traceback.format_exc()

        # Run task evaluation callbacks
        task_evaluation_results = []
        for evaluation in self.current_task.evaluations:
            try:
                result = evaluation["callback"](self)
                task_evaluation_results.append((evaluation['title'], result))
            except Exception as e:
                tb = traceback.format_exc()
                task_evaluation_results.append((evaluation['title'], "ERROR"))

        # TODO: Figure out how I want to allow retries on this phase
        # Run action evaluation callbacks
        action_evaluation_results = []
        for evaluation in self.current_tick.evaluations:
            try:
                result = evaluation["callback"](self)
                action_evaluation_results.append((evaluation['title'], result))
            except Exception as e:
                tb = traceback.format_exc()
                action_evaluation_results.append((evaluation['title'], "ERROR"))
                self.add_error_block("# Evaluation failed: \n"
                                     + f'"""{tb}"""')

        outcomes =  []
        try:
            outcomes += [(self.current_tick.action["title"],action_result),]
        except AttributeError:
            outcomes += [("[No action specified with agent.add_action()]", "ERROR"),]
        outcomes += task_evaluation_results
        outcomes += action_evaluation_results
        
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
        self.debugging = False
        self.failure_stage = "event stream"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", help="The model to use.")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer to use (if different from model_name)")
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
        scores = asyncio.run(evaluate_outputs_vllm(args.model_name,
                                                   score_prompt_fns,
                                                   texts,
                                                   port=args.port))
        return torch.sigmoid(scores)

    def simple_bayes_evaluate_outputs(parent_q, questions, texts):
        if type(texts) == str:
            texts = [texts,]
        score_prompt_fns = [make_simple_bayes_score_prompt(question)
                            for question in questions]
        scores = asyncio.run(bayesian_evaluate_outputs_vllm(args.model_name,
                                                            parent_q,
                                                            score_prompt_fns,
                                                            texts,
                                                            port=args.port))
        return scores

    
    agent = WeaveAgent(args.model_name)

    if not args.tokenizer:
        args.tokenizer = args.model_name

    with open("hf_token.txt") as infile:
        os.environ["HF_TOKEN"] = infile.read().strip()
    # Delete token so it doesn't leak into traces
    os.remove("hf_token.txt")
    agent.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    with open("weave_agent.py") as infile:
        # Genesis block
        genesis_block = {
            'type': 'genesis',
            'body': infile.read()
        }
        agent.add_block(genesis_block)

    with open(args.bootstrap) as infile:
        # Bootstrap block
        bootstrap_block = {
            'type': 'bootstrap',
            'body': infile.read()
        }
        agent.add_block(bootstrap_block)
        exec(bootstrap_block["body"])

    def run_bootstrap_callbacks():
        """Run bootstrap callbacks in function to avoid contaminating global scope."""
        # Run action callback
        action_result = agent.current_tick.action["callback"](agent)

        # Run evaluation callbacks
        evaluation_results = []
        for evaluation in agent.current_tick.evaluations:
            result = evaluation["callback"](agent)
            evaluation_results.append((evaluation['title'], result))

        outcomes =  []
        outcomes += [(agent.current_tick.action["title"],action_result),]
        outcomes += evaluation_results

        # Add outcome block
        outcome_block = {
            'type': 'outcome',
            'table': outcomes
        }
        agent.add_block(outcome_block)
        agent.current_tick.outcome = outcome_block

    run_bootstrap_callbacks()

    # Run the agent
    while True:
        agent.tick()
        time.sleep(1)  # Simulate tick interval
