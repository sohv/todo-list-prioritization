import sys
import os
import matplotlib.pyplot as plt 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.environment import TodoListEnv
from src.dqn_agent import DQNAgent
from src.utils import load_data

# Load data
tasks, user_behavior = load_data()

# Initialize environment and agent
env = TodoListEnv(tasks, user_behavior)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Training parameters
episodes = 1000
batch_size = 64  # Increased batch size
warmup_episodes = 50  # Collect experiences before training

# Create directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)

# Track metrics for plotting
episode_rewards = []
episode_losses = []
avg_rewards = []

# Training loop
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    losses = []
    
    # Update target network periodically
    if e % agent.target_update_freq == 0:
        agent.update_target_model()
        print(f"Target network updated at episode {e}")
    
    for time in range(100):
        # Use epsilon-greedy policy
        if e < warmup_episodes:
            action = np.random.choice(action_size)  # Pure exploration in early episodes
        else:
            action = agent.act(state)
            
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        
        # Store experience
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Train on batch after collecting enough experiences
    if len(agent.memory) > batch_size and e >= warmup_episodes:
        loss = agent.replay(batch_size)
        if loss is not None:
            losses.append(loss)
    
    # Store metrics
    episode_rewards.append(total_reward)
    if losses:
        episode_losses.append(np.mean(losses))
    else:
        episode_losses.append(0)
    
    # Calculate running average of rewards
    if e == 0:
        avg_rewards.append(total_reward)
    else:
        avg_rewards.append(0.9 * avg_rewards[-1] + 0.1 * total_reward)
    
    # Print episode information
    print(f"Episode: {e+1}/{episodes}, Steps: {time+1}, Reward: {total_reward:.2f}, Avg Reward: {avg_rewards[-1]:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    # Save model periodically
    if (e+1) % 50 == 0:
        agent.model.save(f"models/dqn_model_episode_{e+1}.keras")

# Plot training metrics
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episode Rewards')

plt.subplot(1, 3, 2)
plt.plot(avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Running Average Reward')

plt.subplot(1, 3, 3)
plt.plot(episode_losses)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()

# Save final model
agent.model.save("models/dqn_model_final.keras")
print("Training completed and model saved!")