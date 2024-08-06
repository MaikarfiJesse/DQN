import gym
from gym import spaces
import numpy as np
import pygame

class IrrigationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, size=10):
        super().__init__()
        self.size = size
        self.window_size = 800
        self.max_steps = 200

        self.observation_space = spaces.Box(
            low=0, high=4, shape=(size, size), dtype=int
        )
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, -1]),  # Up
            1: np.array([1, 0]),   # Right
            2: np.array([0, 1]),   # Down
            3: np.array([-1, 0])   # Left
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.dry_areas = [(2, 2), (5, 5), (8, 8)]
        self.obstacles = [(3, 3), (6, 6)]
        self.water_source = (0, 0)
        self.current_reward = 0
        self.agent_pos = None
        self.steps_taken = 0

    def _get_obs(self):
        obs = np.zeros((self.size, self.size), dtype=int)
        obs[self.water_source] = 1
        for dry_area in self.dry_areas:
            obs[dry_area] = 2
        for obstacle in self.obstacles:
            obs[obstacle] = 3
        obs[self.agent_pos] = 4
        return obs

    def _get_info(self):
        return {"steps_taken": self.steps_taken}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        self.current_reward = 0
        self.steps_taken = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        new_pos = np.clip(np.array(self.agent_pos) + direction, 0, self.size - 1)

        reward = -0.1  # Default reward for each step

        if tuple(new_pos) in self.dry_areas:
            reward = 10
            done = True
        elif tuple(new_pos) in self.obstacles:
            reward = -5
            done = True
        else:
            self.agent_pos = tuple(new_pos)
            done = False

        self.current_reward += reward
        self.steps_taken += 1

        done = done or self.steps_taken >= self.max_steps

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size + 50)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size + 50))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw the grid and obstacles
        for i in range(self.size):
            for j in range(self.size):
                color = (200, 200, 200)  # Default empty color
                text = ""
                if (i, j) == self.water_source:
                    color = (0, 0, 255)  # Water source
                    text = "Water Source"
                elif (i, j) in self.dry_areas:
                    color = (255, 255, 0)  # Dry area
                    text = "Dry Area"
                elif (i, j) in self.obstacles:
                    color = (255, 0, 0)  # Obstacle
                    text = "Obstacle"
                elif (i, j) == self.agent_pos:
                    color = (0, 255, 0)  # Agent
                    text = "Farmer"
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(j * pix_square_size, i * pix_square_size, pix_square_size, pix_square_size)
                )
                
                # Render text inside the grid cells
                if text:
                    font = pygame.font.SysFont(None, 24)
                    text_surface = font.render(text, True, (0, 0, 0))
                    text_rect = text_surface.get_rect(center=(j * pix_square_size + pix_square_size / 2, i * pix_square_size + pix_square_size / 2))
                    canvas.blit(text_surface, text_rect)

        # Render total reward below the grid
        font = pygame.font.SysFont(None, 40)
        reward_text = f'Total Reward: {self.current_reward:.2f}'
        reward_surface = font.render(reward_text, True, (0, 0, 0))
        canvas.blit(reward_surface, (5, self.window_size + 10))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
