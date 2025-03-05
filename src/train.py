import sys
import os
import matplotlib.pyplot as plt 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.environment import TodoListEnv
from src.dqn_agent import DQNAgent
from src.utils import load_data

def create_directories():
    """Create necessary directories for saving models and plots"""
    directories = ['models', 'plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Load data
tasks, user_behavior = load_data()

# Create necessary directories
create_directories()

# Initialize environment and agent
env = TodoListEnv(tasks, user_behavior)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Training parameters
episodes = 3000
batch_size = 32
save_interval = 100  # Save model every 100 episodes
plot_interval = 50   # Update plot every 50 episodes

# Add these new parameters
warmup_episodes = 100  # Collect experiences before training
min_reward_threshold = -1000  # Minimum acceptable reward
max_steps = 100
validation_interval = 50  # How often to validate performance

# Initialize metrics
episode_rewards = []
running_avg_rewards = []
validation_rewards = []

def validate_agent(env, agent, num_episodes=5):
    """Run validation episodes without exploration"""
    val_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0
        for _ in range(max_steps):
            action = agent.act(state, training=False)  # No exploration
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            episode_reward += reward
            state = next_state
            if done:
                break
        val_rewards.append(episode_reward)
    return np.mean(val_rewards)

# Training loop
best_avg_reward = float('-inf')
no_improvement_count = 0

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    
    # Warmup period: pure exploration
    if e < warmup_episodes:
        agent.epsilon = 1.0
    
    for time in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        
        # Clip extreme rewards
        reward = np.clip(reward, min_reward_threshold, abs(min_reward_threshold))
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Store episode reward
    episode_rewards.append(total_reward)
    
    # Calculate running average
    if len(episode_rewards) > 100:
        avg_reward = np.mean(episode_rewards[-100:])
    else:
        avg_reward = np.mean(episode_rewards)
    running_avg_rewards.append(avg_reward)
    
    # Training
    if len(agent.memory) > batch_size and e >= warmup_episodes:
        agent.replay(batch_size)
    
    # Validation
    if (e + 1) % validation_interval == 0:
        val_reward = validate_agent(env, agent)
        validation_rewards.append(val_reward)
        
        # Early stopping check
        if val_reward > best_avg_reward:
            best_avg_reward = val_reward
            agent.model.save(f"models/dqn_model_best.keras")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # If no improvement for a while, reduce learning rate
        if no_improvement_count >= 5:
            agent.learning_rate *= 0.5
            print(f"Reducing learning rate to {agent.learning_rate}")
            no_improvement_count = 0
    
    # Print progress
    print(f"Episode: {e+1}/{episodes}, Steps: {time+1}, Reward: {total_reward:.2f}, "
          f"Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    # Save model periodically
    if (e + 1) % save_interval == 0:
        model_path = f"models/dqn_model_episode_{e+1}.keras"
        agent.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    # Plot progress periodically
    if (e + 1) % plot_interval == 0:
        plt.figure(figsize=(12, 5))
        
        # Plot episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards')
        
        # Plot running average
        plt.subplot(1, 2, 2)
        plt.plot(running_avg_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward (last 100 episodes)')
        plt.title('Running Average Reward')
        
        plt.tight_layout()
        plt.savefig(f'plots/training_progress_episode_{e+1}.png')
        plt.close()

# Save final model
final_model_path = "models/dqn_model_final.keras"
agent.model.save(final_model_path)
print(f"Training completed! Final model saved to {final_model_path}")

# Create final plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Rewards')

plt.subplot(1, 2, 2)
plt.plot(running_avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward (last 100 episodes)')
plt.title('Running Average Reward')

plt.tight_layout()
plt.savefig('plots/final_training_results.png')
plt.close()