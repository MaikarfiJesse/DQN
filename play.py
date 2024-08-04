from irrigation_env import IrrigationEnv
from tensorflow.keras.models import load_model
from rl.agents import DQNAgent
from rl.policy import GreedyQPolicy

env = IrrigationEnv()
model = load_model('irrigation_model.h5')
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, policy=GreedyQPolicy())
dqn.compile(optimizer='adam', metrics=['mae'])
dqn.load_weights('irrigation_weights.h5f')

# Run some episodes
for episode in range(5):
    state = env.reset()
    done = False
    while not done:
        action = dqn.forward(state)
        state, reward, done, _ = env.step(action)
        env.render()
