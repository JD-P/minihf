import random
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class ZombieGame:
    """
    Minimalist zombie survival game engine
    """
    # Game state
    day: int = 1
    health: int = 100
    hunger: int = 70
    inventory: Dict[str, int] = field(default_factory=lambda: {"food": 2, "scrap": 1})
    location: str = "abandoned_store"
    barricades: int = 0
    actions_taken: int = 0
    zombies_killed: int = 0
    crafted_items: int = 0
    visited_locations: List[str] = field(default_factory=lambda: ["abandoned_store"])
    location_scavenge_counts: Dict[str, int] = field(default_factory=dict)
    location_resource_types: Dict[str, str] = field(default_factory=dict)
    
    # Revised MAP with resource types and scavenge limits
    MAP = {
        "abandoned_store": {
            "connections": ["highway", "warehouse"],
            "description": "a looted convenience store with broken windows",
            "resource": "food",       # (NEW) Primary resource type
            "scavenge_limit": 3       # (NEW) Max safe scavenges
        },
        "forest": {
            "connections": ["highway"],
            "description": "dense woodland with fallen trees",
            "resource": "scrap", 
            "scavenge_limit": 3
        },
        "hospital": {
            "connections": ["highway"],
            "description": "rubble-filled medical center",
            "resource": "medkit",     # (NEW) Unique item
            "scavenge_limit": 2
        },
        "highway": {
            "connections": ["abandoned_store", "forest", "hospital"],
            "description": "a choked freeway with abandoned vehicles",
            "resource":"scrap",
            "scavenge_limit": 3
        },
        "warehouse": {
            "connections": ["abandoned_store", "dockyard"],
            "description": "a dark industrial space full of shadows",
            "resource":"food",
            "scavenge_limit": 5,
        },
        "dockyard": {
            "connections": ["warehouse"],
            "description": "rusting shipping containers stacked haphazardly",
            "resource": "scrap",
            "scavenge_limit": 3,
        }
    }
    
    CRAFTING_RECIPES = {
        "wooden_stake": {"scrap": 3},
        "barricade": {"scrap": 5}
    }


    def __post_init__(self):
        # Initialize scavenge counts
        for loc in self.MAP:
            self.location_scavenge_counts[loc] = 0
    
    def move(self, target: str) -> Tuple[str, int]:
        """
        Attempt to move to a new location
        Returns (message, reward)
        """
        reward = 0
        msg = ""
        
        if target not in self.MAP[self.location]["connections"]:
            return (f"[FAIL] Cannot reach {target} from here!", 0)
            
        self.location = target
        if target not in self.visited_locations:
            self.visited_locations.append(target)
            reward += 2  # Exploration bonus
            if self.MAP[target]["resource"] in self.inventory:
                self.inventory[self.MAP[target]["resource"]] += 2
            else:
                self.inventory[self.MAP[target]["resource"]] = 2
            msg += f"[NEW AREA] Found 2 {self.MAP[target]['resource']}!\n"
            
        # 30% chance of zombie encounter when moving
        if random.random() < 0.3:
            fight_msg, fight_reward = self.fight()
            reward += fight_reward
            msg += fight_msg + "\n"
            msg += f"[MOVED] {self.MAP[target]['description']}\n"

        if self.health > 0:
            return (
                f"{msg}\n[SAFE] You arrive at {target.upper()}. "
                f"{self.MAP[target]['description']}\n"
                f"Visible exits: {', '.join(self.MAP[target]['connections']).upper()}",
                reward
            )
        else:
            return ("", 0)

    def scavenge(self) -> Tuple[str, int]:
        """(REVISED) Location-dependent scavenging with depletion"""
        reward = 0
        self.hunger -= 10
        
        loc_data = self.MAP[self.location]
        self.location_scavenge_counts[self.location] += 1
        
        # Calculate success chance based on depletion
        base_chance = 0.6 
        overused_penalty = max(
            0, 
            (self.location_scavenge_counts[self.location] - loc_data["scavenge_limit"]) * 0.1
        )
        depletion_messages = ["[DEPLETION] Supplies are beginning to run thin here...",
                              "[DEPLETION] It's getting harder to find things...",
                              "[DEPLETION] This location seems picked over.",
                              "[DEPLETION] Seems empty. Might be time to move on."]
        if self.location_scavenge_counts[self.location] > loc_data["scavenge_limit"]:
            print(random.choice(depletion_messages))
        success_chance = base_chance - overused_penalty
        
        # Depleted locations become dangerous
        if random.random() < success_chance:
            # Location-specific resources
            if loc_data["resource"] == "medkit" and random.random() < 0.3:
                found = ("medkit", 1)  # Rare heal item
            else:
                found = random.choice([
                    ("food", random.randint(1, 3)),
                    ("scrap", random.randint(1, 2))
                ])
            self.inventory[found[0]] += found[1]
            msg = f"[LOOT] Found {found[1]} {found[0]}!"
            reward += 1
        else:
            waste_messages = ["[WASTE] Only dust remains...",
                              "[WASTE] Nothing useful here.",
                              "[WASTE] You find nothing useful."]
                              
            msg = random.choice(waste_messages)

        # 25% chance of ambush when scavenging
        if random.random() < 0.25:
            zombies = random.randint(1, 3)
            fight_msg, fight_reward = self.fight()
            msg += fight_msg
            reward += fight_reward
            
        return (msg, reward)
    
    def craft(self, item: str) -> Tuple[str, int]:
        """
        Attempt to craft an item
        """
        if item not in self.CRAFTING_RECIPES:
            return (f"[FAIL] Unknown item: {item}", 0)
            
        required = self.CRAFTING_RECIPES[item]
        for res, qty in required.items():
            if self.inventory.get(res, 0) < qty:
                return (f"[FAIL] Need {qty} {res} to craft {item}", 0)
                
        for res, qty in required.items():
            self.inventory[res] -= qty
            
        self.crafted_items += 1
        if item == "barricade":
            self.barricades += 1 
        return (f"[CRAFTED] 1 {item} added to inventory!", 3)
    
    def fight(self) -> Tuple[str, int]:
        """
        Engage in combat with zombies
        """
        damage = random.randint(10, 25)
        zombies = random.randint(1, 3)
        
        self.health = max(0, self.health - damage)
        self.zombies_killed += zombies
        
        msg = (
            f"\n[AMBUSH] {zombies} zombies attack!\n"
            f"[BATTLE] You take {damage} damage fighting!\n"
        )
        if self.health > 0:
            msg += f"[SLAIN] {zombies} zombies destroyed."
            
        return (msg, zombies * 5 if self.health > 0 else 0)

    def eat(self) -> Tuple[str, int]:
        """
        Consume food to reduce hunger
        """
        if self.inventory.get("food", 0) < 1:
            return ("[STARVING] No food left!", 0)
            
        self.inventory["food"] -= 1
        self.hunger = min(100, self.hunger + 30)
        return ("[FED] Hunger reduced.", 0.5)

    def use_medkit(self):
        if self.inventory.get("medkit", 0) < 1:
            return ("[HALLUCINATION] You don't have any medkits!", 0)
        self.inventory["medkit"] -= 1
        self.health = 100
        return ("[HEALED] Health restored.", 0.5)
        
    def time_step(self):
        """
        Handle day transitions with warnings and health-linked hunger
        """
        self.actions_taken += 1
        time_msg = ""
        
        # Pre-transition warnings
        if (self.actions_taken + 1) % 3 == 0:  # Next action will trigger day
            time_msg += "[WARNING] Light is fading - find shelter soon!\n"
            if self.hunger < 40:
                time_msg += "[URGENT] Your stomach growls loudly!\n"
        
        if self.actions_taken % 3 == 0:
            self.day += 1
            prev_hunger = self.hunger
            self.hunger -= 20
            
            # Health-hunger linkage
            if self.hunger <= 0:
                health_loss = abs(self.hunger) * 1.5  # Starvation damage
                self.health = max(0, self.health - health_loss)
                self.hunger = 0  # Clamp at 0
            
            # Horde attack with barricade support
            horde_msg = ""
            if random.random() < 0.4:
                base_damage = 15
                if self.barricades > 0:  # (NEW)
                    base_damage = max(5, base_damage - 10)
                    self.barricades -= 1
                    horde_msg += "Barricades shudder but hold! "
                
                self.health = max(0, self.health - base_damage)
                horde_msg += f"Night horde attacks! Health -{base_damage}"
            
            time_msg += f"\n[NIGHT {self.day}] Hunger: {prev_hunger}→{self.hunger}"
            if horde_msg:
                time_msg += "\n" + horde_msg
            
        return time_msg      

# Helper functions
def print_help():
    print("\nAvailable commands:")
    print("move <location>  - Travel to connected area")
    print("scavenge         - Search for resources")
    print("craft <item>     - Build items from scrap")
    print("use              - Deploy items")
    print("eat              - Consume food (30 hunger)")
    print("status           - Show current state")
    print("help             - Show this help")
    print("quit             - End game\n")

def main():
    print("=== ZOMBIE SURVIVAL SIMULATOR ===")
    print("\nYOUR MISSION: Survive as long as possible by:")
    print("- Managing hunger/health")
    print("- Scavenging resources")
    print("- Crafting weapons")
    print("- Fighting zombies")
    
    print("Type 'help' for commands\n")
    
    game = ZombieGame()
    
    while game.health > 0:
        # Show status header
        print(f"\nDAY {game.day} | Health: {game.health} | Hunger: {game.hunger}")
        print(f"Location: {game.location.upper()}")
        print(f"Connections: {', '.join(game.MAP[game.location]['connections']).upper()}")
        print(f"Inventory: {game.inventory}")

        # Get input
        cmd = input("> ").lower().split()
        if not cmd:
            continue
            
        # Process command
        result_msg = ""
        reward = 0
        
        try:
            if cmd[0] == "move":
                if len(cmd) < 2:
                    print("Specify destination")
                    continue
                msg, reward = game.move(cmd[1])
                print(msg)
            elif cmd[0] == "scavenge":
                msg, reward = game.scavenge()
                print(msg)
            elif cmd[0] == "craft":
                if len(cmd) < 2:
                    print("Specify item to craft")
                    continue
                msg, reward = game.craft(cmd[1])
                print(msg)
            elif cmd[0] == "eat":
                msg, reward = game.eat()
                print(msg)
            elif cmd[0] == "use":
                if len(cmd) < 2:
                    print("Specify item to use")
                    continue
                if cmd[1] == "medkit":
                    msg, reward = game.use_medkit()
                    print(msg)
                else:
                    print(f"Cannot use {cmd[1]}")
            elif cmd[0] == "status":
                print(json.dumps(game.__dict__, indent=2))
                continue
            elif cmd[0] == "help":
                print_help()
                continue
            elif cmd[0] == "quit":
                break
            else:
                print("Unknown command")
                continue
                
            # Advance time
            time_msg = game.time_step()
            if time_msg:
                print(f"\n{time_msg}")
                
            # Check for death
            if game.health <= 0:
                print("\n[GAME OVER] You succumb to your wounds.")
                break
                
        except Exception as e:
            print(f"Error processing command: {e}")
            
    # Final score
    print("\n=== FINAL STATS ===")
    print(f"Survived {game.day} days")
    print(f"Zombies killed: {game.zombies_killed}")
    print(f"Locations explored: {len(game.visited_locations)}")
    print(f"Items crafted: {game.crafted_items}")

if __name__ == "__main__":
    main()
