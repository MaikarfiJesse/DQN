#!/usr/bin/env python3
import streamlit as st
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from irrigation_env import IrrigationEnv
import time

def play_simulation():
    # Initialize the Irrigation environment
    env = DummyVecEnv([lambda: IrrigationEnv()])

    # Load the trained model
    model = DQN.load('dqn_irrigation_model')

    # Streamlit setup
    st.title('Irrigation Agent Simulation')
    stframe = st.empty()

    # Run the agent in the environment
    obs = env.reset()
    done = False
    step = 0
    max_steps = 5 * 60 * 20  # 5 minutes at 20 fps

    while not done and step < max_steps:
        # Predict the action based on the current observation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        # Render the environment
        img = env.render(mode='rgb_array')
        stframe.image(img, caption=f'Step: {step}', use_column_width=True)

        # Increment step counter
        step += 1

        # Pause to control the speed of the simulation
        time.sleep(0.05)

    st.write("Simulation complete.")

if __name__ == "__main__":
    play_simulation()
