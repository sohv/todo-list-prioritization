import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import TodoListEnv
from dqn_agent import DQNAgent

# Set the device to MPS if available
device_name = '/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
print(f"Using device: {device_name}")

def create_directories():
    """Create necessary directories for saving models and plots"""
    directories = ['models', 'plots', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def train_model(
    train_data_dir="data/train",
    episodes=4000,
    batch_size=128,
    save_interval=100,
    plot_interval=50,
    max_steps_per_episode=100
):
    """
    Train the DQN agent on the training dataset with status-aware tracking.
    """
    print("Starting training process...")
    
    # Create necessary directories
    create_directories()
    
    try:
        # Load training data
        print("Loading training data...")
        train_tasks = pd.read_csv(os.path.join(train_data_dir, 'tasks.csv'))
        train_behavior = pd.read_csv(os.path.join(train_data_dir, 'user_behavior.csv'))
        print(f"Loaded {len(train_tasks)} training tasks")
        
        # Print training data statistics
        print("\nTraining Data Statistics:")
        print("\nTask Status Distribution:")
        print(train_tasks['status'].value_counts(normalize=True))
        print("\nPriority Distribution:")
        print(train_tasks['priority'].value_counts())
        print("\nDeadline Range:")
        print(f"Earliest: {train_tasks['deadline'].min()}")
        print(f"Latest: {train_tasks['deadline'].max()}")
        
        # Initialize environment
        print("\nInitializing environment...")
        env = TodoListEnv(train_tasks, train_behavior)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(f"Environment initialized with state size: {state_size}, action size: {action_size}")
        
        # Initialize agent
        print("\nInitializing agent...")
        agent = DQNAgent(state_size, action_size)
        
        # Training metrics
        episode_rewards = []
        running_avg_rewards = []
        status_based_rewards = {
            'completed': [],
            'in_progress': [],
            'todo': []
        }
        
        # Open log file
        log_file = open("logs/training_log.txt", "w")
        
        # Training loop
        print("\nStarting training loop...")
        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            episode_status_rewards = {
                'completed': 0,
                'in_progress': 0,
                'todo': 0
            }
            
            for time in range(max_steps_per_episode):
                # Get action
                action = agent.act(state)
                
                # Get task status
                task_status = train_tasks.iloc[action]['status']
                
                # Take step
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                
                # Track reward by status
                episode_status_rewards[task_status] += reward
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Train on batch
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            # Store metrics
            episode_rewards.append(total_reward)
            for status, reward in episode_status_rewards.items():
                status_based_rewards[status].append(reward)
            
            # Calculate running average
            if len(episode_rewards) > 100:
                avg_reward = np.mean(episode_rewards[-100:])
            else:
                avg_reward = np.mean(episode_rewards)
            running_avg_rewards.append(avg_reward)
            
            # Log progress
            log_message = (
                f"Episode: {e+1}/{episodes}\n"
                f"Steps: {time+1}, Total Reward: {total_reward:.2f}\n"
                f"Running Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}\n"
                f"Status Rewards: completed={episode_status_rewards['completed']:.2f}, "
                f"in_progress={episode_status_rewards['in_progress']:.2f}, "
                f"todo={episode_status_rewards['todo']:.2f}\n"
                f"------------------------\n"
            )
            print(log_message)
            log_file.write(log_message)
            
            # Save model periodically
            if (e + 1) % save_interval == 0:
                model_path = f"models/dqn_model_episode_{e+1}.keras"
                agent.model.save(model_path)  # Save as .keras format
                print(f"Model saved to {model_path}")
            
            # Plot progress periodically
            if (e + 1) % plot_interval == 0:
                plt.figure(figsize=(15, 5))
                
                # Plot episode rewards
                plt.subplot(1, 3, 1)
                plt.plot(episode_rewards)
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.title('Episode Rewards')
                
                # Plot running average
                plt.subplot(1, 3, 2)
                plt.plot(running_avg_rewards)
                plt.xlabel('Episode')
                plt.ylabel('Average Reward (last 100 episodes)')
                plt.title('Running Average Reward')
                
                # Plot status-based rewards
                plt.subplot(1, 3, 3)
                for status, rewards in status_based_rewards.items():
                    plt.plot(rewards, label=status)
                plt.xlabel('Episode')
                plt.ylabel('Reward by Status')
                plt.title('Status-based Rewards')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f'plots/training_progress_episode_{e+1}.png')
                plt.close()
        
        # Save final model
        final_model_path = "models/dqn_model_final.keras"
        agent.model.save(final_model_path)
        print(f"\nTraining completed! Final model saved to {final_model_path}")
        
        # Close log file
        log_file.close()
        
        # Create final plots
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Rewards')
        
        plt.subplot(1, 3, 2)
        plt.plot(running_avg_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Running Average Reward')
        
        plt.subplot(1, 3, 3)
        for status, rewards in status_based_rewards.items():
            plt.plot(rewards, label=status)
        plt.xlabel('Episode')
        plt.ylabel('Reward by Status')
        plt.title('Status-based Rewards')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/final_training_results.png')
        plt.close()
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model(
        train_data_dir="data/train",
        episodes=4000,
        batch_size=128,
        save_interval=100,
        plot_interval=50,
        max_steps_per_episode=100
    )