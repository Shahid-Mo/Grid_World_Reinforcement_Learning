import gym
from gym import spaces
import numpy as np

class StochasticGridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.grid = np.array([
            [0, 1, 0, 0, 3],
            [0, 1, 0, 2, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 4]
        ])
        self.num_rows, self.num_cols = self.grid.shape
        self.num_states = self.num_rows * self.num_cols
        self.num_actions = 4
        
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)
        
        self.state = None
        self.done = False
        self.reward_range = (-50, 20)
        self.step_count = 0
        self.reset()
        
    def reset(self):
        self.state = 0  # Start from the top right corner
        self.done = False
        self.step_count = 0
        return self.state
    
    def step(self, action):
        reward = -1
        next_state = self.state
        row, col = np.divmod(self.state, self.num_cols)
        if np.random.rand() < 0.1:  # 10% chance of choosing a random action
            action = np.random.randint(self.num_actions)
            #print('random action')
            #print(action)
            
        # if action is chosen as up, probability of selecting up is 0.9*1 +0.1*0.25 = 0.925
        
        if action == 0:  # Up
            next_state = max(0, row - 1) * self.num_cols + col
        elif action == 1:  # Down
            next_state = min(self.num_rows - 1, row + 1) * self.num_cols + col
        elif action == 2:  # Left
            next_state = row * self.num_cols + max(0, col - 1)
        elif action == 3:  # Right
            next_state = row * self.num_cols + min(self.num_cols - 1, col + 1)
        
        if next_state == 1:  # If the current state is a reward state
            reward = -50
        elif next_state == 6:  # If the current state is a penalty state
            reward = -50
        elif next_state == 4:  # If the current state is a reward state
            reward = 0
        elif next_state == 8:
            reward = -2
        elif next_state == 13:  # If the current state is a penalty state
            reward = -50
        elif next_state == 18:  # If the current state is a reward state
            reward = -50
        elif next_state == 19:
            reward = 20
        
        self.state = next_state
        self.step_count += 1 
        if self.state==19 or self.state ==4 or self.step_count >= 100:  # If the current state is a terminal state
            self.done = True
        
        return next_state, reward, self.done, {}
    

    def render(self):
        for i in range(self.num_rows):
            row_str = ""
            for j in range(self.num_cols):
                if (i * self.num_cols + j) == self.state:  # Agent
                    row_str += "A "
                elif self.grid[i, j] == 4:  # Terminal 1
                    row_str += "T "
                elif self.grid[i, j] == 3:  # Terminal 2
                    row_str += "T "
                elif self.grid[i, j] == 1:  # Obstacle
                    row_str += "# "
                elif self.grid[i, j] == 2:
                    row_str += "* "
                elif self.grid[i, j] == 0:  
                    row_str += ". "
            print(row_str)
        print()