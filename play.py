import gym
from stable_baselines3 import DQN
from irrigation_env import IrrigationEnv
import pygame
import time
import numpy as np

# Define colors
COLOR_WATER_SOURCE = (0, 0, 255)
COLOR_DRY_AREA = (255, 255, 0)
COLOR_OBSTACLE = (255, 0, 0)
COLOR_AGENT = (0, 255, 0)
COLOR_EMPTY = (200, 200, 200)

def render_env(env, screen, block_size=40):
    # Get the observation
    obs = env._get_observation()
    field_size = obs.shape[0]
    
    for i in range(field_size):
        for j in range(field_size):
            color = COLOR_EMPTY
            if obs[i, j] == 1:
                color = COLOR_WATER_SOURCE
            elif obs[i, j] == 2:
                color = COLOR_DRY_AREA
            elif obs[i, j] == 3:
                color = COLOR_OBSTACLE
            elif obs[i, j] == 4:
                color = COLOR_AGENT
            pygame.draw.rect(screen, color, pygame.Rect(j * block_size, i * block_size, block_size, block_size))
    pygame.display.flip()

def play_simulation():
    # Initialize pygame
    pygame.init()
    block_size = 40
    field_size = 10  # Adjust according to your environment
    screen_size = (field_size * block_size, field_size * block_size)
    screen = pygame.display.set_mode(screen_size)
    
    # Initialize the environment
    env = IrrigationEnv(size=field_size)
    
    # Load the trained model
    model = DQN.load("dqn_irrigation")
    
    # Reset the environment
    obs = env.reset()
    done = False
    total_reward = 0
    start_time = time.time()
    max_time = 300  # 5 minutes in seconds

    while (time.time() - start_time) < max_time:
        if done:
            # Restart the environment if done
            obs = env.reset()
            done = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Predict the action based on the current observation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        
        # Accumulate reward
        total_reward += reward
        
        # Render the environment
        render_env(env, screen, block_size)
        
        # Delay to ensure smooth animation
        time.sleep(1)

    # Print results
    print(f"Total Reward: {total_reward}")
    print(f"Simulation Done: {done}")

    # Close the environment
    pygame.quit()

if __name__ == "__main__":
    play_simulation()
