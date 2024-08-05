import json
import random
import time
import types
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

# Adapted from the following two JSON grammars, both apache licensed like MiniHF
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/guided_decoding/outlines_decoding.py#L31
# https://github.com/outlines-dev/outlines/blob/main/outlines/grammars/json.lark
json_lines_grammar = r"""
?start: json_lines

json_lines: json_line (NEWLINE json_line)*

json_line: object | array

?value: object
| array
| UNESCAPED_STRING
| SIGNED_NUMBER      -> number
| "true"             -> true
| "false"            -> false
| "null"             -> null

array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : UNESCAPED_STRING ":" value

%import common.UNESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%import common.NEWLINE

%ignore WS
"""
        
class WeaveAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.event_stream = []
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
            json.dump(out, outfile)
        raise SystemExit
        
    def add_block(self, block):
        block['index'] = self.current_block_index
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

    def render_context(self):
        self.context = ""
        for event_block in self.event_stream[-32:]:
            self.context += (json.dumps(event_block) + "\n")
        return self.context

    def generate_block(self, block_type, context):
        """Generate a block and add it to the event stream."""
        prompt = f'{context}{{"type": "{block_type}"'
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
                   "guided_grammar":json_lines_grammar,
                   "stop":["\n"]}
        response = requests.post(f"http://localhost:{port}/v1/completions/",
                                 data=json.dumps(payload))
        texts = [choice["text"] for choice in response.json()["choices"]]
        if texts:
            block = json.loads(
                f'{{"type": "{block_type}"'
                + ","
                + texts[0].strip()[1:]
            )
            self.add_block(block)
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
            elif reminder['trigger_type'] == 'callback':
                if reminder['trigger_callback'](self) >= reminder['threshold']:
                    reminder['reminder_callback'](self)
        
    def tick(self):
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
        observation_blocks = [{'type': 'observation', 'content': view['callback'](self)} for view in self.observation_views]

        # Inject these into the event stream
        self.event_stream += (task_blocks + observation_blocks)
            
        # Render context
        self.render_context()

        # Write orientation reasoning block
        # This is your opportunity to analyze the situation based on the
        # observation, reminder, task, etc blocks. Use this moment to decide
        # what to do next.
        self.generate_block("orientation", self.context)
        self.render_context()
        
        # Write action block
        action_block = self.generate_block("action", self.context)
        self.render_context()
        # Write evaluation programs
        for _ in range(3):
            self.generate_block("evaluation", self.context)
            self.render_context()

        with open("confirm.txt", "w") as outfile:
            print(self.event_stream[-5:])
            json.dump(self.event_stream[-5:], outfile)
        while os.path.exists("confirm.txt"):
            time.sleep(1)
            
        # Execute action program
        try:
            exec(action_block['program'])
        except Exception as e:
            self.add_error_block(f"Action execution failed: {e}")
            return

        # Run evaluation programs if action did not fail
        for block in self.event_stream:
            if block['type'] == 'evaluation':
                try:
                    exec(block['program'])
                except Exception as e:
                    self.add_error_block(f"Evaluation execution failed: {e}")

# Example usage
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", help="The model to use.")
    parser.add_argument("--port", default=5000, help="The port to use for VLLM.")
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

    with open("bootstrap.py") as infile:
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
