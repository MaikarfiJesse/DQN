from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from irrigation_env import IrrigationEnv

# Create environment
env = DummyVecEnv([lambda: IrrigationEnv()])

# Define the model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("dqn_irrigation")

# Load the model
model = DQN.load("dqn_irrigation")

# Play simulation
def play_simulation():
    env = DummyVecEnv([lambda: IrrigationEnv(render_mode="human")])
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    play_simulation()
