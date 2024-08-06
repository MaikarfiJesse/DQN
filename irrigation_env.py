import gym
from gym import spaces
import numpy as np
import pygame 
class IrrigationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, size=5):
        super().__init__()
        self.size = size
        self.window_size = 800
        self.max_steps = 100

        self.observation_space = spaces.Box(low=0, high=1, shape=(size * size + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # Actions: 0: No Irrigation, 1: Low, 2: Medium, 3: High

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.reset()

    def reset(self):
        self.state = np.zeros(self.size * self.size)
        self.water_level = 0.5  # Example: initial water level
        self.step_count = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs()

    def step(self, action):
        if action == 1:
            self.water_level = min(1.0, self.water_level + 0.1)
        elif action == 2:
            self.water_level = min(1.0, self.water_level + 0.3)
        elif action == 3:
            self.water_level = min(1.0, self.water_level + 0.5)
        
        reward = self._compute_reward()
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.state, [self.water_level]])

    def _compute_reward(self):
        return -np.abs(self.water_level - 0.5)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()  # Initialize pygame here
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pygame.draw.rect(canvas, (0, 0, 255), (self.window_size * self.water_level, self.window_size * 0.5, 50, 50))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            import pygame.surfarray
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
