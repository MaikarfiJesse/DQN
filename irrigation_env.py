import gym
import numpy as np
from gym import spaces

class IrrigationEnv(gym.Env):
    def __init__(self, size=10):
        super(IrrigationEnv, self).__init__()
        
        # Define the size of the field
        self.field_size = size

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Actions: Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.field_size, self.field_size), dtype=np.float32)
        
        # Define irrigation goals and obstacles
        self.water_source = (0, 0)
        self.dry_areas = [(2, 2), (5, 5), (8, 8)]
        self.obstacles = [(3, 3), (6, 6)]
        
        # Initialize agent position
        self.agent_pos = None
        
        # Set maximum steps
        self.max_steps = 200
        self.current_step = 0

    def reset(self):
        self.agent_pos = (np.random.randint(0, self.field_size), np.random.randint(0, self.field_size))
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        
        # Move agent based on action
        if action == 0:  # Up
            self.agent_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
        elif action == 1:  # Down
            self.agent_pos = (min(self.field_size - 1, self.agent_pos[0] + 1), self.agent_pos[1])
        elif action == 2:  # Left
            self.agent_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
        elif action == 3:  # Right
            self.agent_pos = (self.agent_pos[0], min(self.field_size - 1, self.agent_pos[1] + 1))
        
        # Check for termination conditions
        done = False
        reward = -0.1  # Small negative reward for each step
        
        if self.agent_pos in self.dry_areas:
            reward = 10
            done = True
        elif self.agent_pos in self.obstacles:
            reward = -5
            done = True
        elif self.current_step >= self.max_steps:
            done = True
        
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        obs = np.zeros((self.field_size, self.field_size), dtype=np.float32)
        obs[self.water_source] = 1
        for dry_area in self.dry_areas:
            obs[dry_area] = 2
        for obstacle in self.obstacles:
            obs[obstacle] = 3
        obs[self.agent_pos] = 4
        return obs

    def render(self, mode='human'):
        if mode == 'human':
            for i in range(self.field_size):
                for j in range(self.field_size):
                    if (i, j) == self.agent_pos:
                        print('🌾', end=' ')
                    elif (i, j) == self.water_source:
                        print('💧', end=' ')
                    elif (i, j) in self.dry_areas:
                        print('🌵', end=' ')
                    elif (i, j) in self.obstacles:
                        print('🛑', end=' ')
                    else:
                        print('.', end=' ')
                print()
            print('\n')
