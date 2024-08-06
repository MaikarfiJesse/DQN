import gym
from stable_baselines3 import DQN
from irrigation_env import IrrigationEnv
import pygame
import numpy as np

def simulate():
    # Initialize Pygame
    pygame.init()
    screen_size = 800
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption('Irrigation Simulation')

    # Load the trained model
    model = DQN.load("dqn_irrigation")

    # Initialize the environment
    env = IrrigationEnv(size=10)

    # Initialize variables for rendering
    clock = pygame.time.Clock()
    total_reward = 0

    # Simulate irrigation
    obs = env.reset()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Predict action
        action, _states = model.predict(obs)

        # Take action in the environment
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Render environment (custom rendering)
        screen.fill((255, 255, 255))  # Clear screen with white background

        # Draw the environment
        # Assuming env.render() has been replaced with Pygame drawing logic
        env_drawn = env.render(mode='rgb_array')
        surface = pygame.surfarray.make_surface(np.transpose(env_drawn, (1, 0, 2)))
        screen.blit(surface, (0, 0))

        # Draw total reward
        font = pygame.font.SysFont(None, 36)
        reward_text = font.render(f'Total Reward: {total_reward:.2f}', True, (0, 0, 0))
        screen.blit(reward_text, (10, 10))

        pygame.display.flip()
        clock.tick(30)  # Limit frames per second

    pygame.quit()

if __name__ == "__main__":
    simulate()
