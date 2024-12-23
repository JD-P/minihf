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
from copy import deepcopy
from pprint import pformat
from argparse import ArgumentParser
from typing import List, Dict, Optional, Any
from jsonschema import validate
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

class WeaveAgentTask:
    def __init__(self, subagent, title: str, description: str = ""):
        self.subagent = subagent
        self.title = str(title)
        self.description = description
        self.evaluations = []

    def add_evaluation(self, title, callback):
        assert type(title) == str
        assert type(callback) == types.FunctionType
        self.evaluations.append({"type":"evaluation",
                                 "title":title,
                                 "callback":callback})

    def run_evaluations(self):
        results = {}
        for evaluation in self.evaluations:
            try:
                result = evaluation["callback"](self.subagent)
            except Exception as e:
                result = traceback.format_exc()
            results[evaluation["callback"].__name__] = result
        return results

# Earlier versions of the weave-agent used a flat chain of code blocks that manage
# problem state by interacting with a global kanban board. The idea was that each
# sub-task in the agents overall goal could be represented as a card on the board
# and then the agent sets the current task, flags tasks that have been blocked or
# turned out to be based on invalid premises, etc. There were multiple problems
# with this that the data structure below solves to create a more coherent problem
# solving strategy. The first was that the agent wouldn't remember to manage the
# content of the kanban board without explicit prompting, which led to adding a
# whole stage in its core loop dedicated just to doing so called task-inference.
# Task-inference didn't have a set expected structure and took place before action,
# which meant that it became possible for the agent to get itself stuck in a loop
# of trying to resolve a task over and over. Another problem was that the agent
# would often try to resolve a task prematurely, so it became necessary to add
# unit and sanity tests that have to be satisfied before a task can be marked
# completed. This limited the ability of the agent to set its own tasks and
# break problems into parts. A third problem was that the flow control when
# a task was blocked and should be returned to its parent was janky and had to
# be performed manually.
#
# The WeaveAgentTree was inspired by watching an instance of the weave-agent try
# to write an action block with subroutines and asking "that strategy it wanted
# to try looks pretty good, but the framework doesn't provide the affordance for
# it to try it, it runs out of space in the length limit on actions before it
# finishes and assumes subroutines are there that don't exist, how could I make
# this pattern natural for it?". What I realized was that if I gave up on the
# idea of being able to change goals in the middle of a task that having an
# expected type of return value and a series of steps to achieve it was similar
# to a function call. We could reformulate the weave-agent then as a call tree
# of subagents that are given a task with predefined conditions checked against
# a data structure returned by the subagent. To help encourage good habits
# correctness is checked at multiple levels. Perhaps the most important problem
# the WeaveAgentTree solves is planning: Writing programs with subroutines
# is a form of hierarchical planning that's in distribution for any code model.
# Because the task structure is now built into the call tree there's a smooth
# natural abstraction telling the weave-agent when to formulate goals, when the
# goals are completed, how to check it did them right, where to put the results,
# and how to transfer control of execution once it's finished. All of these
# operations go from being awkward conscious affairs to smooth unconscious
# bodily structure.
    
class WeaveAgentTree:
    def __init__(self, model_name: str, time_budget: int):
        self.model_name = model_name
        self.__agents = {}
        self.__time_budget = time_budget
        # Pin genesis and bootstrap so agent knows how to use framework
        self.__pinned_events = [0, 1]
        self.__current_block_index = 0
        self.__event_stream = []

    def run(self, name):
        import time
        start_time = time.time()
        deadline = float(self.__agents[name].end_time)
        return_schema = deepcopy(self.__agents[name].schema)
        result = self.__agents[name].run()
        validate(instance=result, schema=return_schema)
        end_time = time.time()
        if end_time > deadline + 300:
            # TODO: More nuanced way to handle this
            raise ValueError("Time exceeded!")
        else:
            return result
        
    def subagent(self, name, parent, description, schema, time_budget):
        if name in self.__agents:
            raise ValueError
        reserved_words = {"name", "description", "children", "schema"}
        assert not set(schema).intersection(reserved_words)
        if parent:
            self.__agents[parent].children.append(name)
        try:
            subagent = WeaveAgentNode(self, parent, name, description, schema, time_budget)
        except Exception as e:
            self.__agents[parent].children.remove(name)
            raise e
        self.__agents[name] = subagent
        return subagent

    def add_block(self, block):
        block['index'] = self.__current_block_index
        block['timestamp'] = time.time()
        if block['type'] == 'orientation':
            block['metadata'] = {
                "block_index":self.__current_block_index,
                "working_directory":os.getcwd()
            }
        if "q" not in block:
            block["q"] = ""
        if "score" not in block:
            #TODO: Make actual score function for observations, task reminders etc
            block["score"] = 2
        if "tags" not in block:
            #TODO: Make actual tagging function
            block["tags"] = ["placeholder",]
        self.__event_stream.append(block)
        
        if block["type"] not in {"genesis", "bootstrap"}:
            writer = bm25_index.writer()
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
        
        self.__current_block_index += 1

    def current_block_index(self):
        return self.__current_block_index
            
    def render_context(self):
        context = ""
        context_blocks = []
        history_len = 60
        for index in self.__pinned_events:
            if (len(self.__event_stream) - index) > history_len:
                context_blocks.append(self.__event_stream[index])
        context_blocks += self.__event_stream[-history_len:]
        for event_block in context_blocks:
            context += render_block(event_block)
        return context
        
    def view_board(self, root="main") -> str:
        problem_map = {}
        substack = [root,]
        while substack:
            subagent = self.__agents[substack.pop()]
            parent = subagent.name
            path = []
            while parent:
                path.append(parent)
                # Convert to object so we can get grandparent
                parent = self.__agents[parent]
                parent = parent.parent
            path.reverse()
            current_level = problem_map
            for key in path:
                if key not in current_level:
                    current_level[key] = {}
                current_level = current_level[key]
            current_level["name"] = subagent.name
            current_level["description"] = subagent.task.description
            current_level["evaluations"] = subagent.task.run_evaluations()
            current_level["time_remaining"] = subagent.end_time - time.time()
            current_level["completed"] = subagent.completed
            current_level["schema"] = subagent.schema
            substack.extend(subagent.children)
        return pformat(problem_map)

    def dump_event_stream(self):
        with open(f"/app/weave-agent-logs/event_trace_{round(time.time())}.json", "w") as outfile:
            json.dump(self.__event_stream, outfile)
        with open(f"/app/weave-agent-logs/rendered_trace_{round(time.time())}.py", "w") as outfile:
            for event_block in self.__event_stream:
                outfile.write(render_block(event_block))
            outfile.flush()


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
        

# The intended problem solving strategy for subagents is to delegate until you
# reach a base case that can be solved in a short number of actions and then
# resolve it. The root task is allocated a certain amount of time which it can
# then delegate to subagent calls. Remember not to allocate all of the available
# time to a call tree unless you're very rushed, you should assume there will be
# failures and budget tasks the time that they need rather than just splitting
# up the available time between them.
    
class WeaveAgentNode:
    def __init__(self, tree, parent, subagent_name, description, schema, time_budget):
        self.tree = tree
        self.parent = parent
        self.children = []
        self.model_name = self.tree.model_name
        self.name = subagent_name
        self.schema = schema
        self.creation_time = time.time()
        self.time_budget = time_budget
        self.end_time = self.creation_time + (time_budget * 60)
        self.current_tick = Tick(self, 0)
        self.ticks = []
        self.debugging = False
        self.failure_stage = "event stream"
        self.task = WeaveAgentTask(self, self.name, description)
        self.observation_views = []
        # TODO: Do I really need to have this pointer?
        self.bm25_index = bm25_index
        self.tools = {}
        self.cache = {}
        self.context = ""
        self.completed = False

    def run(self):
        """Run the subagent."""
        self.start_time = time.time()
        self.end_time = self.start_time + (self.time_budget * 60)
        while (time.time() < self.end_time) and not self.completed:
            self.tick()
            time.sleep(1)
        return self.completed

    # TODO: Assert that subagent unit test callbacks have names before adding them
    def return_to_caller(self, value: dict):
        """Return thread of execution from subagent to caller. This should be 
        called when the agent's task has been resolved, the task is deemed 
        intractable, or the agent has wandered off so far it can't find 
        its way back to the task."""
        value["name"] = self.name
        value["description"] = self.task.description
        value["children"] = self.children
        schema["name"] = "string"
        schema["description"] = "string"
        schema["children"] = "list"
        schema["schema"] = "object"
        for callback_name, result in self.task.run_evaluations():
            value[callback_name] = result
            self.schema[callback_name] = {"type": ["boolean", "integer", "float"]}
        value["schema"] = self.schema
        validate(instance=value, schema=self.schema)
        # Setting this interrupts the inference loop and signals an exit
        self.completed = value

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
        self.context = self.tree.render_context()

    def generate_block(self, block_type, context, eval_questions, weave_params, hint=""):
        """Generate a block and add it to the event stream."""
        return generate_block_inner(self, block_type, context, eval_questions, weave_params, hint)

    def add_block(self, block):
        block["subagent"] = self.name
        block["time_remaining"] = self.end_time - time.time()
        self.tree.add_block(block)
    
    def add_error_block(self, error_message):
        self.debugging = True
        error_block = {
            'type': 'error',
            'message': error_message
        }
        self.add_block(error_block)
                    
    def tick(self):
        self.tree.dump_event_stream()
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
            # if self.current_task:
                # TODO: Figure out how to bind evaluation definitions to task
                # so that the agent can be reminded of how the unit tests are
                # defined exactly and therefore what is expected.
                #task_reminder_body += "# Current Task:\n"
                #task_reminder_body += ('"""\n' + self.task.view_task() + '\n"""\n')
            task_reminder_body += "# Problem Map:\n"
            task_reminder_body += ('"""\n' + self.tree.view_board() + '\n"""')
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
        for new_block in (task_blocks + observation_blocks):
            self.add_block(new_block)
            
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
            rprint(f"Writing block #[cyan]{self.tree.current_block_index()}[/cyan] of type [cyan]{block_type}[/cyan]")
            try:
                block = self.generate_block(block_type,
                                            self.context,
                                            inference_questions,
                                            weave_params,
                                            hint=hint)
            except ValueError as e:
                tb = traceback.format_exc()
                # TODO: This isn't even correct, replace with dynamic hints -_-
                hint = ("Hint: callbacks are structured like\n\n"
                        + "def callback_name(subagent):\n   "
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
            + "# monologue in a triple quote block at this step.\n"
            + "# Orientation uses the MiniHF Morpheus format. Morpheus\n"
            + "# solves problems through discussion between personas\n"
            + "# or postures representing different aspects of weave-agent\n"
            + "# such as mental motions, perspectives on the problem, etc.\n"
            + "# The first posture is always expectation because at the\n"
            + "# start of a tick we evaluate whether the expectation we\n"
            + "# formed about the action taken in the last tick was\n"
            + "# violated or not. The different personas then discuss\n"
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
        )
        mcts_params = {"weave_n_tokens":256, "weave_budget":288,
                       "weave_round_budget":96, "weave_n_expand":32}
        orientation_block = do_tick_block(self,
                                          "orientation",
                                          orientation_hint,
                                          mcts_params)
        if orientation_block:
            self.current_tick.orientation = orientation_block
        else:
            return
        
        # Write action block
        action_hint = (
            "#hint Action blocks are where I write code to take actions.\n"
            + "# If the task makes sense to break into parts, define subagents\n"
            + "# to delegate to using agent.subagent(). Make sure to define a\n"
            + "# schema and task evaluations for each subagent. If it won't fit\n"
            + "# into one action block keep in mind you can define subagents \n"
            + "# across multiple blocks and then do agent.run() to execute them.\n"
            + "# If it seems possible to resolve the current task as a base case\n"
            + "# in a handful of actions then write a callback to further my goal(s)\n"
            + "# based on the orientation block and set up the callback to be\n" 
            + "# executed with the agent.add_action() method. I must write a \n"
            + "# callback and then set it up to be executed\n"
            + "# later with agent.add_action() or the tick will not be accepted.\n"
            + "# It's important to remember that my callback can do anything\n"
            + "# a python program can do through side effects in the external\n" 
            + "# computable environment. If I need to import a new module make sure\n"
            + "# to do it inside the callback because the tick gets executed in a\n"
            + "# local context."
        )
        for i in range(3):
            action_block = do_tick_block(self,
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

            # Run action callback
            try:
                action_result = self.current_tick.action["callback"](self)
            except Exception as e:
                action_result = traceback.format_exc()
                tb = action_result
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
        expectation_block = do_tick_block(self,
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
            + "# next tick. I should avoid redundant observation callbacks. I\n"
            + "# can remove ones that are no longer necessary or mostly distracting\n"
            + "# with remove_observation_view(view_title). If new callbacks seem useful\n"
            + "# to help me orient and judge whether the action had the intended\n"
            + "# side effects on the computable environment I can add them\n"
            + "# with add_observation_view(title, callback)"
        )
        observation_inference_block = do_tick_block(self,
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
                eval_block = do_tick_block(self,
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

        # Run task evaluation callbacks
        task_evaluation_results = []
        for evaluation in self.task.evaluations:
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
    parser.add_argument("--budget", type=int, default=360,
                        help="Time budget for the run in minutes.")
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

    
    agent = WeaveAgentTree(args.model_name, args.budget)

    if not args.tokenizer:
        args.tokenizer = args.model_name

    with open("hf_token.txt") as infile:
        os.environ["HF_TOKEN"] = infile.read().strip()
    # Delete token so it doesn't leak into traces
    os.remove("hf_token.txt")
    agent.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    schema_builder = SchemaBuilder()
    schema_builder.add_text_field("type", stored=True)
    schema_builder.add_text_field("render", stored=True)
    schema_builder.add_text_field("q", stored=True)
    schema_builder.add_float_field("score", stored=True)
    schema_builder.add_integer_field("index", stored=True)
    schema_builder.add_float_field("timestamp", stored=True)
    schema_builder.add_text_field("tags", stored=True)

    bm25_schema = schema_builder.build()

    if not os.path.exists("memories"):
        os.mkdir("memories")
    if not os.path.exists("memories/bm25"):
        os.mkdir("memories/bm25")
    bm25_index = Index(bm25_schema, path="./memories/bm25")
    
    # Mock bootstrap agent so we can run the callbacks in bootstrap file
    self = agent.subagent(
        "bootstrap",
        None,
        "Bootstrap the weave-agent",
        {},
        args.budget,
                          
    )
    with open("weave_agent.py") as infile:
        # Genesis block
        genesis_block = {
            'type': 'genesis',
            'body': infile.read()
        }
        self.add_block(genesis_block)

    with open(args.bootstrap) as infile:
        # Bootstrap block
        bootstrap_block = {
            'type': 'bootstrap',
            'body': infile.read()
        }
        self.add_block(bootstrap_block)
        exec(bootstrap_block["body"])

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

    if not os.path.exists("/app/weave-agent-logs"):
        os.mkdir("/app/weave-agent-logs")
        
    result, event_stream = agent.run("main")
    
    with open(f"/app/weave-agent-logs/{round(time.time())}/log.json", "w") as outfile:
        out = {"model_name":args.model_name,
               "event_stream":event_stream,
               "result":result,}
        json.dump(out, outfile)
        outfile.flush()
