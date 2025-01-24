#subagent main
#startblock type: evaluation
#index 42
#timestamp 1737568442.5842712
#time_remaining 20415.256756067276 seconds

def have_enough_hitpoints(subagent):
    import re
    """Check if we have enough hitpoints after drinking the potion."""
    screen = subagent.tools['nethack'].pane.capture_pane()
    pattern = r"HP:(\d+)\((\d+)\)"
    match = re.search(pattern, screen_text)
    current_hp = int(match.group(1))
    max_hp = int(match.group(2))
    if current_hp < 10:
        return 0
    else:
        return 1
    
self.add_evaluation('Check enough hitpoints', have_enough_hitpoints)
#q: Is this evaluation a good implementation of a callback to gather and evaluate the expected sensory evidence laid out in the expectation block? Yes. (84.375%)
#endblock
