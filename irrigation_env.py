import gym
import numpy as np
from gym import spaces
import pygame

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

        # Initialize pygame
        pygame.init()
        self.screen_size = 500
        self.cell_size = self.screen_size // self.field_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Irrigation Simulation")

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
            # Fill the screen with white
            self.screen.fill((255, 255, 255))

            # Draw the grid
            for i in range(self.field_size):
                for j in range(self.field_size):
                    rect = pygame.Rect(i * self.cell_size, j * self.cell_size, self.cell_size, self.cell_size)
                    if (i, j) == self.agent_pos:
                        pygame.draw.rect(self.screen, (0, 255, 0), rect)  # Agent in green
                    elif (i, j) == self.water_source:
                        pygame.draw.rect(self.screen, (0, 0, 255), rect)  # Water source in blue
                    elif (i, j) in self.dry_areas:
                        pygame.draw.rect(self.screen, (255, 255, 0), rect)  # Dry area in yellow
                    elif (i, j) in self.obstacles:
                        pygame.draw.rect(self.screen, (255, 0, 0), rect)  # Obstacle in red
                    else:
                        pygame.draw.rect(self.screen, (200, 200, 200), rect)  # Empty cell in light gray
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # Draw grid lines

            pygame.display.flip()
