import gym
import numpy as np
from gym import spaces

class IrrigationEnv(gym.Env):
    def __init__(self):
        super(IrrigationEnv, self).__init__()

        # Define the size of the grid (for simplicity, using a 5x5 grid)
        self.grid_size = 5

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=5, shape=(self.grid_size, self.grid_size), dtype=np.float32)

        # Define positions of static objects
        self.water_source = (4, 4)
        self.crops = [(0, 1), (2, 1), (4, 1), (1, 3), (3, 3)]
        self.rock = (3, 2)
        
        # Initialize agent position
        self.agent_pos = None

        # Set maximum steps
        self.max_steps = 100
        self.current_step = 0

    def reset(self):
        # Reset agent to a random position
        self.agent_pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        self.current_step += 1

        # Move agent based on action
        if action == 0:  # Up
            self.agent_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
        elif action == 1:  # Down
            self.agent_pos = (min(self.grid_size - 1, self.agent_pos[0] + 1), self.agent_pos[1])
        elif action == 2:  # Left
            self.agent_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
        elif action == 3:  # Right
            self.agent_pos = (self.agent_pos[0], min(self.grid_size - 1, self.agent_pos[1] + 1))

        # Check for termination conditions
        done = False
        reward = -0.1  # Small negative reward for each step

        if self.agent_pos == self.water_source:
            reward = 10
            done = True
        elif self.agent_pos == self.rock:
            reward = -5
            done = True
        elif self.agent_pos in self.crops:
            reward += 1
        elif self.current_step >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        obs[self.agent_pos] = 1
        obs[self.water_source] = 2
        for crop in self.crops:
            obs[crop] = 3
        obs[self.rock] = 4
        return obs

    def render(self, mode='human'):
        if mode == 'human':
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if (i, j) == self.agent_pos:
                        print('🙂', end=' ')
                    elif (i, j) == self.water_source:
                        print('💧', end=' ')
                    elif (i, j) in self.crops:
                        print('🌱', end=' ')
                    elif (i, j) == self.rock:
                        print('🪨', end=' ')
                    else:
                        print('.', end=' ')
                print()
            print('\n')
