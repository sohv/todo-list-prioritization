import sys
import os
import matplotlib.pyplot as plt 

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.environment import TodoListEnv
from src.dqn_agent import DQNAgent
from src.utils import load_data

# Load data
tasks, user_behavior = load_data()

# Initialize environment and agent
env = TodoListEnv(tasks, user_behavior)
state_size = env.observation_space.shape[0]  # Use shape[0] for flattened state size
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Training parameters
episodes = 3000
batch_size = 32

# Track rewards for plotting
episode_rewards = []

# Training loop
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    
    for time in range(100):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if done:
            print(f"Episode: {e}/{episodes}, Score: {time}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            episode_rewards.append(total_reward)
            break
        
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    
    # Save the model
    if e % 10 == 0:
        agent.model.save("models/dqn_model.h5")

# Plot rewards
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()