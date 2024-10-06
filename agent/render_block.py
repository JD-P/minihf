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

def render_block(event_block):
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
        return (header + "\n" + event_block["program"] + footer)
    elif event_block["type"] == "bootstrap":
        footer += "# END OF DEMO. Starting on the next tick you have\n"
        footer += "# full control. Wake up.\n"
        return (header + "\n" + event_block["program"] + footer)
    elif event_block["type"] == "observation":
        title = event_block['title']
        if type(event_block['content']) != str:
            content = repr(event_block['content'])
        else:
            content = event_block['content']
        header += f"#title {title}\n"
        lines = content.split('\n')
        content = ""
        for line in lines:
            content += f"# {line}\n"
        return (header + "\n" + content + footer)
    elif event_block["type"] == "orientation":
        metadata = event_block["metadata"]
        header += f"# Starting tick #{metadata['tick_number']} "
        header += f"with block #{metadata['block_index']}\n"
        header += f"# Current Working Directory: {metadata['working_directory']}\n"
        return (header + "\n" + event_block["program"] + footer)
    elif event_block["type"] == "task_inference":
        metadata = event_block["metadata"]
        task_title = f"({metadata['task_id']}) {metadata['task_title']}"
        task_status = f"({metadata['task_status']}) {metadata['task_explanation']}"
        header += f"# Current Task: {task_title}\n"
        header += f"# Task Status: {task_status}\n"
        return (header + "\n" + event_block["program"] + footer)
    elif event_block["type"] == "task-reminder":
        return (header + "\n" + event_block["task"] + footer)
    elif event_block["type"] == "error":
        header += "# WARNING: Error means tick did not fully execute callbacks\n"
        return (header + "\n" + event_block["message"] + footer)
    elif event_block["type"] == "outcome":
        return (header + "\n" 
                         + generate_outcome_table(event_block['table'])
                         + footer)
    else:
        return (header + repr(event_block) + footer)
