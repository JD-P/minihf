import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("trace", help="The JSON of the event blocks from the weave-agent.")
args = parser.parse_args()

with open(args.trace) as infile:
    events = json.load(infile)

def generate_outcome_table(evaluation_results):
    table = "Evaluation Results:\n"
    table += "--------------------\n"
    for program, result in evaluation_results:
        table += f"Program: {program}\n"
        table += f"Result: {result}\n"
        table += "--------------------\n"
    return table

context = ""
for event_block in events:
    header = f'#startblock type: {event_block["type"]}\n'
    if "timestamp" in event_block:
        header += f'#timestamp {event_block["timestamp"]}\n\n'
    footer = '\n#endblock\n'
    if event_block["type"] in ("genesis",
                               "task_inference",
                               "orientation",
                               "action",
                               "expectation",
                               "observation_inference",
                               "evaluation"):
        context += (header + event_block["program"] + footer)
    elif event_block["type"] == "bootstrap":
        footer += "# End of Demo. Starting on the next tick you have\n"
        footer += "# full control. Wake up.\n"
        context += (header + event_block["program"] + footer)
    elif event_block["type"] == "task-reminder":
        context += (header + event_block["task"] + footer)
    elif event_block["type"] == "error":
        header += "# WARNING: ERROR MEANS TICK DID NOT FINISH EXECUTION\n"
        header += "# ADDRESS ERROR IN NEXT TICK BEFORE PROCEEDING\n"
        context += (header + event_block["message"] + footer)
    elif event_block["type"] == "outcome":
        context += (header
                         + generate_outcome_table(event_block['table'])
                         + footer)
    else:
        context += (header + repr(event_block) + footer)

print(context)
