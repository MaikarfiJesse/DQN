import gym
import numpy as np
from stable_baselines3 import DQN
from irrigation_env import IrrigationEnv
import pygame
import time
import random

# Define colors
COLOR_WATER_SOURCE = (0, 0, 255)
COLOR_DRY_AREA = (255, 255, 0)
COLOR_OBSTACLE = (255, 0, 0)
COLOR_AGENT = (0, 255, 0)
COLOR_EMPTY = (200, 200, 200)
COLOR_TEXT = (0, 0, 0)

def render_env(env, screen, font, block_size=60):
    # Get the observation
    obs = env._get_observation()
    field_size = obs.shape[0]
    
    for i in range(field_size):
        for j in range(field_size):
            color = COLOR_EMPTY
            text = ""
            if obs[i, j] == 1:
                color = COLOR_WATER_SOURCE
                text = "Water Source"
            elif obs[i, j] == 2:
                color = COLOR_DRY_AREA
                text = "Dry Area"
            elif obs[i, j] == 3:
                color = COLOR_OBSTACLE
                text = "Obstacle"
            elif obs[i, j] == 4:
                color = COLOR_AGENT
                text = "Farmer"
            pygame.draw.rect(screen, color, pygame.Rect(j * block_size, i * block_size, block_size, block_size))
            
            # Render text
            if text:
                text_surface = font.render(text, True, COLOR_TEXT)
                text_rect = text_surface.get_rect(center=(j * block_size + block_size / 2, i * block_size + block_size / 2))
                screen.blit(text_surface, text_rect)
    pygame.display.flip()

def generate_varied_actions(env, duration=300, interval=0.5):
    actions = []
    start_time = time.time()
    current_time = 0

    while current_time < duration:
        current_time = time.time() - start_time
        actions.append(random.choice([0, 1, 2, 3]))  # Randomly choose between Up, Down, Left, Right
        time.sleep(interval)

    return actions

def play_simulation():
    # Initialize pygame
    pygame.init()
    block_size = 60
    field_size = 10  # Adjust according to your environment
    screen_size = (field_size * block_size, field_size * block_size)
    screen = pygame.display.set_mode(screen_size)
    
    # Set font for text rendering
    font = pygame.font.SysFont(None, 28)
    
    # Initialize the environment
    env = IrrigationEnv(size=field_size)
    
    # Load the trained model
    model = DQN.load("dqn_irrigation")
    
    # Generate action sequence
    action_sequence = generate_varied_actions(env)

    # Reset the environment
    obs = env.reset()
    total_reward = 0
    start_time = time.time()
    max_time = 300  # 5 minutes in seconds

    for action in action_sequence:
        if (time.time() - start_time) >= max_time:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Execute the action
        obs, reward, done, _ = env.step(action)
        
        # Accumulate reward
        total_reward += reward
        
        # Render the environment
        render_env(env, screen, font, block_size)
        
        # Delay to ensure smooth animation
        time.sleep(0.5)

        if done:
            obs = env.reset()

    # Print results
    print(f"Total Reward: {total_reward}")

    # Close the environment
    pygame.quit()

if __name__ == "__main__":
    play_simulation()
