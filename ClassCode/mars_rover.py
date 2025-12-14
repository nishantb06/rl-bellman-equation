import numpy as np
import random
import os

# Action Constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

class MarsRoverEnv:
    def __init__(self, size=6, slippery=False):
        self.size = size
        self.slippery = slippery  # True = MDP (Stochastic), False = Deterministic
        
        # Grid Setup
        self.agent_pos = (0, 0)
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        
        # Hazards (Static for now, but can be randomized)
        # Using a set for O(1) lookups
        self.pits = { (1, 3), (2, 3), (3, 3), (4, 1) }

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        """
        Executes one time step within the environment.
        Returns: next_state, reward, done
        """
        # 1. Determine the actual movement direction
        if self.slippery:
            # 80% chance success, 10% slip left, 10% slip right relative to direction
            move_direction = self._get_stochastic_direction(action)
        else:
            move_direction = action

        # 2. Calculate tentative new position
        row, col = self.agent_pos
        d_row, d_col = 0, 0
        
        if move_direction == UP:    d_row = -1
        elif move_direction == DOWN:  d_row = 1
        elif move_direction == LEFT:  d_col = -1
        elif move_direction == RIGHT: d_col = 1
        
        new_row = row + d_row
        new_col = col + d_col

        # 3. Check Boundaries (If you hit a wall, you stay put)
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            self.agent_pos = (new_row, new_col)
        
        # 4. Check Rewards and Terminal States
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 10.0, True  # +10 for Water Ice
        
        if self.agent_pos in self.pits:
            return self.agent_pos, -10.0, True # -10 for Crater
            
        # Standard step cost (Living Penalty)
        # Encourages finding the shortest path
        return self.agent_pos, -0.1, False

    def _get_stochastic_direction(self, action):
        """
        Simulates slippery wheels. 
        If trying to go UP: 0.8 -> UP, 0.1 -> LEFT, 0.1 -> RIGHT
        """
        if np.random.rand() < 0.8:
            return action
        else:
            # Pick one of the perpendicular directions randomly
            if action in [UP, DOWN]:
                return np.random.choice([LEFT, RIGHT])
            else:
                return np.random.choice([UP, DOWN])

    def render(self):
        """Simple text render for manual play"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Mars Rover Status: {'SLIPPERY' if self.slippery else 'STABLE'}")
        print("-" * (self.size * 4 + 1))
        
        for r in range(self.size):
            line = "|"
            for c in range(self.size):
                pos = (r, c)
                content = "   "
                if pos == self.agent_pos:
                    content = " R " # Rover
                elif pos == self.goal_pos:
                    content = " G " # Goal
                elif pos in self.pits:
                    content = " X " # Pit
                line += content + "|"
            print(line)
            print("-" * (self.size * 4 + 1))


# ==========================================
# Manual Play Loop
# ==========================================
if __name__ == "__main__":
    # Change slippery to True to test Phase 2 mechanics
    env = MarsRoverEnv(size=6, slippery=False)
    state = env.reset()
    done = False
    
    print("Controls: W (Up), S (Down), A (Left), D (Right)")
    input("Press Enter to start...")

    while not done:
        env.render()
        user_input = input("Action (WASD): ").upper()
        
        action = None
        if user_input == 'W': action = UP
        elif user_input == 'S': action = DOWN
        elif user_input == 'A': action = LEFT
        elif user_input == 'D': action = RIGHT
        
        if action is not None:
            state, reward, done = env.step(action)
            print(f"Reward: {reward}")
        
        if done:
            env.render()
            if reward > 0:
                print("SUCCESS! You found water!")
            else:
                print("CRASH! Rover destroyed.")