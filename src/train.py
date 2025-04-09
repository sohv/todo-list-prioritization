import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from environment import TodoListEnv
from dqn_agent import DQNAgent

# use Apple GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# model.to(device)
# data = data.to(device)

print(f"Using device: {device}")

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
    
    create_directories()
    
    try:
        print("Loading training data...")
        train_tasks = pd.read_csv(os.path.join(train_data_dir, 'tasks.csv'))
        train_behavior = pd.read_csv(os.path.join(train_data_dir, 'user_behavior.csv'))
        print(f"Loaded {len(train_tasks)} training tasks")
        
        print("\nTraining Data Statistics:")
        print("\nTask Status Distribution:")
        print(train_tasks['status'].value_counts(normalize=True))
        print("\nPriority Distribution:")
        print(train_tasks['priority'].value_counts())
        print("\nDeadline Range:")
        print(f"Earliest: {train_tasks['deadline'].min()}")
        print(f"Latest: {train_tasks['deadline'].max()}")
        
        print("\nInitializing environment...")
        env = TodoListEnv(train_tasks, train_behavior)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(f"Environment initialized with state size: {state_size}, action size: {action_size}")
        
        print("\nInitializing agent...")
        agent = DQNAgent(state_size, action_size, device=device)
        
        # training metrics
        episode_rewards = []
        running_avg_rewards = []
        status_based_rewards = {
            'completed': [],
            'in_progress': [],
            'todo': []
        }
        
        log_file = open("logs/training_log.txt", "w")
        
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
                action = agent.act(state)
                task_status = train_tasks.iloc[action]['status']
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                
                episode_status_rewards[task_status] += reward
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward 
                if done:
                    break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            episode_rewards.append(total_reward)
            for status, reward in episode_status_rewards.items():
                status_based_rewards[status].append(reward)
            
            if len(episode_rewards) > 100:
                avg_reward = np.mean(episode_rewards[-100:])
            else:
                avg_reward = np.mean(episode_rewards)
            running_avg_rewards.append(avg_reward)
            
            log_message = ( # log progress
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
            
            if (e + 1) % save_interval == 0:
                model_path = f"models/dqn_model_episode_{e+1}.pt"
                agent.save(model_path)
                print(f"Model saved to {model_path}")
            
            if (e + 1) % plot_interval == 0:
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.plot(episode_rewards)
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.title('Episode Rewards')
                
                plt.subplot(1, 3, 2)
                plt.plot(running_avg_rewards)
                plt.xlabel('Episode')
                plt.ylabel('Average Reward (last 100 episodes)')
                plt.title('Running Average Reward')
                
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
        
        final_model_path = "models/dqn_model_final.pt"
        agent.save(final_model_path)
        print(f"\nTraining completed! Final model saved to {final_model_path}")
        
        log_file.close()
        
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