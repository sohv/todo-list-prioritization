# script to evaluate the trained agent
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.environment import TodoListEnv
from src.dqn_agent import DQNAgent
from src.utils import load_data

tasks, user_behavior = load_data()

# Initialize environment and agent
env = TodoListEnv(tasks, user_behavior)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Load the trained model
agent.model.load_weights("models/dqn_model.h5")

# Evaluate the agent
state = env.reset()
state = np.reshape(state, [1, state_size])
total_reward = 0

for time in range(100):
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, state_size])
    total_reward += reward
    state = next_state

    if done:
        print(f"Evaluation completed. Total Reward: {total_reward}")
        break