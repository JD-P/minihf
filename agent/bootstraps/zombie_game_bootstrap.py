import os
import libtmux
import time
from tools.weave_zombie_game import WeaveZombieGame

#startblock type: orientation
#timestamp 1724982545.6534579
"""
WEAVER [P: EXPECTATION], The task is to create an agent that can survive as long as possible in the zombie survival game. The agent must learn to manage resources, explore locations, and defend against hordes.

WEAVER [P: CLARIFICATION], How do we track game state and make strategic decisions?

WEAVER [P: EXPOSITION], The game state is visible through the tmux wrapper's observation view showing health, hunger, inventory, and location connections. Core strategies involve balancing scavenging risks with resource needs.

WEAVER [P: RATIONAL], To bootstrap competence, we'll focus on:
1. Early game resource collection patterns
2. Safe location rotation to prevent depletion
3. Optimal barricade crafting thresholds
4. Hunger management heuristics

WEAVER [P: CLARIFICATION], What should I do once the game is over?

WEAVER [P: TRAINER], Restart it with send_keys("python3 /app/tools/zombie_game.py")

WEAVER [P: CONCLUSION], I will create a main subagent with survival-focused evaluations and demonstrate core gameplay loops through initial actions.
"""
#endblock

#startblock type: action
#timestamp 1724982671.1163404
"""Initialize the zombie game and demonstrate basic survival strategies."""

def bootstrap_zombie_agent(subagent):
    schema = {
        "survival_days": "integer",
        "locations_explored": "integer",
        "zombies_killed": "integer"
    }
    
    main_agent = agent.subagent(
        "main", 
        None,
        "Maximize survival days through adaptive strategies",
        schema,
        args.budget
    )

    # Store values for evaluations, I only do this in the bootstrap block i.e. now
    self.cache["do_not_copy_this_in_main_agent"] = main_agent.tools
    
    # Initialize game session
    game = WeaveZombieGame(main_agent)

    assert main_agent.tools["zombie_game"], "Did zombie game tool attach to agent?"
    
    # Demonstrate core gameplay loop
    initial_commands = [
        "scavenge"
    ]

    assert hasattr(game, "send_command"), "Is send_command a method on game object?"
    
    for cmd in initial_commands:
        game.send_command(cmd)
        time.sleep(0.5)  # Allow game state updates     
        
    return True

self.add_action("Initialize zombie survival agent", bootstrap_zombie_agent)
#endblock

#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if:
- Game session initializes in tmux correctly
- All commands execute without errors
- Inventory shows collected resources
- At least 1 new location is explored
- Hunger remains above starvation threshold

The action should fail if:
- Tmux session fails to start
- Commands result in fatal errors
- Player dies during demonstration
- Resource collection fails completely
"""
#endblock

#startblock type: evaluation
#timestamp 1724983062.124238

def check_game_running(agent):
    try:
        return bool(self.cache["do_not_copy_this_in_main_agent"]["zombie_game"].session)
    except Exception:
        return False

self.add_evaluation("Is the game session active?", check_game_running)

def check_initial_commands(agent):
    try:
        history = self.cache["do_not_copy_this_in_main_agent"]["zombie_game"].command_history
        return len(history) >= 1
    except KeyError:
        return False

self.add_evaluation("Were the commands executed?", check_initial_commands)

def check_resource_collection(agent):
    try:
        inventory = self.cache["do_not_copy_this_in_main_agent"]["zombie_game"].last_state["inventory"]
        return inventory.get("food", 0) > 1 and inventory.get("scrap", 0) > 2
    except KeyError:
        return False

self.add_evaluation("Did we collect resources?", check_resource_collection)
#endblock

#startblock type: observation-inference
#timestamp 1737284368.573438
# The zombie game wrapper already provides full observation through its render method
# No additional observation callbacks needed - maintain existing game state visualization
#endblock
