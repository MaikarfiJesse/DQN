import gym
from stable_baselines3 import DQN
# from irrigation_env import IrrigationEnv

def simulate():
    # Load the trained model
    model = DQN.load("dqn_irrigation")

    # Initialize the environment
    env = IrrigationEnv(size=10)

    # Simulate irrigation
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()

if __name__ == "__main__":
    simulate()
