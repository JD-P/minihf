def bootstrap_callback(agent):
    # Set up a reminder
    agent.add_reminder({
        'type': 'reminder',
        'trigger_callback': lambda agent: simple_evaluate_outputs([make_simple_score_prompt("Is it time to remind the agent?"),], agent.context),
        'reminder_callback': lambda agent: agent.add_block({'type': 'reminder', 'message': 'This is a reminder'}),
        'trigger_type': 'yes_no_logit',
        'question': 'Is it time to remind the agent?',
        'threshold': 0.5
    })

    # Set up an observation view that reads from scratch.txt
    def read_scratch_file(agent):
        try:
            with open('scratch.txt', 'r') as file:
                content = file.read()
            return content
        except Exception as e:
            agent.add_error_block(f"Failed to read scratch.txt: {e}")
            return ""

    agent.add_observation_view({
        'type': 'observation',
        'callback': read_scratch_file
    })

bootstrap_callback(agent)
