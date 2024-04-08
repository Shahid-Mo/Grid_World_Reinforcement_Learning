import gym
from gym import spaces
import numpy as np

class SimpleGridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.grid = np.array([
            [0, 1, 0, 0, 3],
            [0, 1, 0, 2, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 4]
        ])
        self.num_rows, self.num_cols = self.grid.shape # there are 12 states
        self.num_states = self.num_rows * self.num_cols
        self.num_actions = 4 # 4 actions up, Down, Left, Right
        
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)
        
        self.state = None
        self.done = False
        self.reward_range = (-50, 10) #5 rewards -1, -50, -5 , 0 , 10 
        self.step_count = 0
        self.reset()
        
    def reset(self):
        self.state = 0  # Start from the top right corner
        self.done = False
        self.step_count = 0
        return self.state
    
    def step(self, action):
        reward = -1 # negative reward for every timestep (want agent to make efficient moves)
        next_state = self.state
        row, col = np.divmod(self.state, self.num_cols)
        
        if action == 0:  # Up
            next_state = max(0, row - 1) * self.num_cols + col
        elif action == 1:  # Down
            next_state = min(self.num_rows - 1, row + 1) * self.num_cols + col
        elif action == 2:  # Left
            next_state = row * self.num_cols + max(0, col - 1)
        elif action == 3:  # Right
            next_state = row * self.num_cols + min(self.num_cols - 1, col + 1)
        
        # Agent is not allowed to go out of bounds
        
        if next_state == 1:  # Max Penalty 
            reward = -50
        elif next_state == 6:  # Max Penalty
            reward = -50
        elif next_state == 4:  # Terminal state 1 (want the agent to learn to dodge it)
            reward = 0
        elif next_state == 8:  # Small penalty (want agen to take this)
            reward = -2
        elif next_state == 13:  # Max Penalty
            reward = -50
        elif next_state == 18:  # Max Penalty
            reward = -50
        elif next_state == 19:  #Terminal state 2 (want the agent to learn to go to it)
            reward = 20
        
        self.state = next_state
        self.step_count += 1 
        if self.state==19 or self.state ==4 or self.step_count >= 50:  # If the current state is a terminal state
            self.done = True
        
        return next_state, reward, self.done, {}
    

    def render(self):
        for i in range(self.num_rows):
            row_str = ""
            for j in range(self.num_cols):
                if (i * self.num_cols + j) == self.state:  # Agent
                    row_str += "A "
                elif self.grid[i, j] == 4:  # Terminal 1
                    row_str += "T1 "
                elif self.grid[i, j] == 3:  # Terminal 2
                    row_str += "T2 "
                elif self.grid[i, j] == 1:  
                    row_str += "# "
                elif self.grid[i, j] == 2:
                    row_str += "* "
                elif self.grid[i, j] == 0:  
                    row_str += ". "
            print(row_str)
        print()
