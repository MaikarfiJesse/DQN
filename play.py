import gym
from stable_baselines3 import DQN
from irrigation_env import IrrigationEnv

def simulate():
    env = IrrigationEnv(size=10)
    obs = env.reset()
    done = False
    
    while not done:
        # Render the environment (text-based)
        env.render()
        
        # Sample a random action
        action = env.action_space.sample()
        
        # Take a step in the environment
        obs, reward, done, info = env.step(action)

        # Add a delay for better visualization (optional)
        import time
        time.sleep(1)
    
    # Final render
    env.render()
    env.close()

if __name__ == "__main__":
    simulate()
