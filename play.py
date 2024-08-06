#!/usr/bin/env python3
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from irrigation_env import IrrigationEnv  # Adjust import if necessary
import numpy as np
import time

def play_simulation():
    # Initialize the environment
    env = DummyVecEnv([lambda: IrrigationEnv(render_mode="human", size=5)])

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

        # Flatten the observation array before prediction
        obs = np.squeeze(obs)  # Flatten the observation to match the expected shape

        # Predict the action based on the current observation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Accumulate reward
        total_reward += reward[0]

        # Render the environment
        env.render()

        # Delay to ensure smooth animation
        time.sleep(1 / env.metadata["render_fps"])

    # Print results
    print(f"Total Reward: {total_reward}")
    print(f"Simulation Done: {done}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    play_simulation()
