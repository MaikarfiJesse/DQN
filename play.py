import gym
from stable_baselines3 import DQN
from irrigation_env import IrrigationEnv

def main():
    env = IrrigationEnv(size=10)
    obs = env.reset()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, _ = env.step(action)
        if done:
            break
    env.close()

if __name__ == "__main__":
    main()
