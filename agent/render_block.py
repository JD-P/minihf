import math
import random
import torch

def generate_outcome_table(evaluation_results):
    table = "Evaluation Results:\n"
    table += "--------------------\n"
    for program, result in evaluation_results:
        table += f"Program: {program}\n"
        table += f"Result: {result}\n"
        table += "--------------------\n"
    return table

def render_block(event_block, tags=True):
    defunct_body_keys = {
        "genesis":"program",
        "bootstrap":"program",
        "observation":"content",
        "orientation":"program",
        "task-inference":"program",
        "action":"program",
        "expectation":"program",
        "observation-inference":"program",
        "evaluation":"program",
        "task-reminder":"task",
        "error":"message"
    }
    if "body" not in event_block and event_block["type"] in defunct_body_keys:
        event_block["body"] = event_block[defunct_body_keys[event_block["type"]]]
    
    header = ""
    if "subagent" in event_block:
        header += f'#subagent {event_block["subagent"]}\n'
    header += f'#startblock type: {event_block["type"]}\n'
    if "index" in event_block:
        header += f'#index {event_block["index"]}\n'
    if "timestamp" in event_block:
        header += f'#timestamp {event_block["timestamp"]}\n'
    if "time_remaining" in event_block:
        header += f'#time_remaining {event_block["time_remaining"]} seconds\n'
    if "block_size" in event_block:
        if event_block["block_size"] == "full":
            header += f'#block_size I have 768 tokens (full) to write with\n'
        elif event_block['block_size'] == "half":
            header += f'#block_size I have 384 tokens (half) to write with\n'
        elif event_block['block_size'] == "quarter":
            header += f'#block_size I have 192 tokens (quarter) to write with\n'
        else:
            header += f'#block_size [INVALID BLOCKSIZE GIVEN] (How did that happen?)'
    if "bm25_query" in event_block:
        header += f'#bm25_query {event_block["bm25_query"]}\n'
    footer = ""
    if "score" in event_block and event_block["q"]:
        score = event_block["score"]
        if score == float('inf'):
            score = 9
        elif math.isnan(score):
            score = 0
        yes_p = torch.sigmoid(torch.tensor(score)).item()
        no_p = 1 - yes_p
        yes_p, no_p = round(yes_p, 5), round(no_p, 5)
        answer = random.choices(["Yes.", "No."], weights=[yes_p, no_p])[0]
        if answer == "Yes.":
            prob = f"({round(yes_p * 100, 5)}%)"
        else:
            prob = f"({round(no_p * 100, 5)}%)"
        footer += f'\n#q: {event_block["q"]} {answer} {prob}'
    if "tags" in event_block:
        footer += f"\n#tags: {' '.join(event_block['tags'])}"
    if "outcome" in event_block:
        with open("run_without_errors_questions.txt") as infile:
            question = random.choice(infile.readlines())
        if event_block["outcome"]["error"]:
            footer += f"\n#q: {question.strip()} No."
        else:
            footer += f"\n#q: {question.strip()} Yes."
    footer += '\n#endblock\n'
    if event_block["type"] in ("genesis",
                               "action",
                               "expectation",
                               "observation-inference",
                               "evaluation",
                               "debug",
                               "backtrack"):
        return (header + "\n" + event_block["body"] + footer)
    elif event_block["type"] == "bootstrap":
        footer += "# END OF DEMO. Starting on the next tick you have\n"
        footer += "# full control. Wake up.\n"
        return (header + "\n" + event_block["body"] + footer)
    elif event_block["type"] == "observation":
        title = event_block['title']
        if type(event_block['body']) != str:
            content = repr(event_block['body'])
        else:
            content = event_block['body']
        header += f"#title {title}\n"
        lines = content.split('\n')
        content = ""
        for line in lines:
            content += f"# {line}\n"
        return (header + "\n" + content + footer)
    elif event_block["type"] == "orientation":
        if "metadata" in event_block:
            metadata = event_block["metadata"]
            header += f"# Starting new tick "
            header += f"with block #{metadata['block_index']}\n"
            header += f"# Current Working Directory: {metadata['working_directory']}\n"
        return (header + "\n" + event_block["body"] + footer)
    elif event_block["type"] == "task-inference":
        if "metadata" in event_block:
            metadata = event_block["metadata"]
            task_title = f"({metadata['task_id']}) {metadata['task_title']}"
            task_status = f"({metadata['task_status']}) {metadata['task_explanation']}"
            header += f"# Current Task: {task_title}\n"
            header += f"# Task Status: {task_status}\n"
        return (header + "\n" + event_block["body"] + footer)
    elif event_block["type"] == "task-reminder":
        return (header + "\n" + event_block["body"] + footer)
    elif event_block["type"] == "error":
        header += "# WARNING: Error means last callback was not fully executed\n"
        return (header + "\n" + event_block["body"] + footer)
    elif event_block["type"] == "outcome":
        return (header + "\n" 
                         + generate_outcome_table(event_block['table'])
                         + footer)
    elif event_block["type"] == "option":
        # Ignore everything else and just do a simple render
        score = event_block["score"]
        if score == float('inf'):
            score = 9
        elif math.isnan(score):
            score = 0
        yes_p = torch.sigmoid(torch.tensor(score)).item()
        no_p = 1 - yes_p
        yes_p, no_p = round(yes_p, 5), round(no_p, 5)
        if event_block["body"] == "Yes.":
            prob = f"({round(yes_p * 100, 5)}%)"
        else:
            prob = f"({round(no_p * 100, 5)}%)"
        return f"#q: {event_block['q']} {event_block['body']} {prob}\n"
    else:
        return (header + repr(event_block) + footer)
