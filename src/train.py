import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from .environment import TodoListEnv
from .dqn_agent import DQNAgent

# use NVIDIA GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_directories():
    """Create necessary directories for saving models and plots"""
    directories = ['models', 'plots', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def validate_agent(agent, val_env, num_episodes=10):
    """Run validation episodes and return average performance"""
    total_scores = []
    
    for _ in range(num_episodes):
        state, _ = val_env.reset()
        total_reward = 0
        
        while True:
            valid_actions = val_env.get_valid_actions()
            if not valid_actions:
                break
                
            # Use agent in evaluation mode (no exploration)
            action = agent.act(state, valid_actions, training=False)
            state, reward, done, truncated, _ = val_env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
                
        total_scores.append(total_reward)
    
    return np.mean(total_scores)

def create_training_plots(episode_rewards, episode_lengths, loss_history, 
                        validation_scores, tasks_completed, episode_num, final=False):
    """Create comprehensive training plots"""
    suffix = "final" if final else f"episode_{episode_num}"
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True)
    
    # Running average reward
    window = min(100, len(episode_rewards))
    if len(episode_rewards) >= window:
        running_avg = [np.mean(episode_rewards[max(0, i-window):i+1]) 
                      for i in range(len(episode_rewards))]
        axes[0, 1].plot(running_avg)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Running Average Reward')
    axes[0, 1].set_title(f'Running Average Reward (window={window})')
    axes[0, 1].grid(True)
    
    # Episode lengths
    axes[0, 2].plot(episode_lengths)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Episode Length')
    axes[0, 2].set_title('Episode Lengths (Tasks Completed)')
    axes[0, 2].grid(True)
    
    # Loss history
    if loss_history:
        axes[1, 0].plot(loss_history)
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Loss Data', ha='center', va='center')
        axes[1, 0].set_title('Training Loss')
    
    # Validation scores
    if validation_scores:
        val_episodes = np.arange(0, len(validation_scores)) * 200  # Assuming validation every 200 episodes
        axes[1, 1].plot(val_episodes, validation_scores, 'ro-')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Validation Score')
        axes[1, 1].set_title('Validation Performance')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Validation Data', ha='center', va='center')
        axes[1, 1].set_title('Validation Performance')
    
    # Tasks completed distribution
    if tasks_completed:
        axes[1, 2].hist(tasks_completed, bins=20, alpha=0.7)
        axes[1, 2].set_xlabel('Tasks Completed per Episode')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Tasks Completed Distribution')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/training_progress_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_model(
    train_data_dir="data/train",
    test_data_dir="data/test", 
    episodes=1000,
    batch_size=128,
    plot_interval=50,
    validation_interval=200,
    early_stopping_patience=500
):
    """
    Train the DQN agent with proper episode management and validation.
    """
    print("Starting training process...")
    
    create_directories()
    
    try:
        print("Loading training data...")
        train_tasks = pd.read_csv(os.path.join(train_data_dir, 'tasks.csv'))
        train_behavior = pd.read_csv(os.path.join(train_data_dir, 'user_behavior.csv'))
        print(f"Loaded {len(train_tasks)} training tasks")
        
        # Load test data if available
        test_tasks = None
        test_behavior = None
        if os.path.exists(test_data_dir):
            try:
                test_tasks = pd.read_csv(os.path.join(test_data_dir, 'tasks.csv'))
                test_behavior = pd.read_csv(os.path.join(test_data_dir, 'user_behavior.csv'))
                print(f"Loaded {len(test_tasks)} test tasks for validation")
            except:
                print("Test data not found, skipping validation")
        
        print("\nTraining Data Statistics:")
        print(f"Task Status Distribution:")
        print(train_tasks['status'].value_counts(normalize=True))
        print(f"\nPriority Distribution:")
        print(train_tasks['priority'].value_counts())
        print(f"\nDeadline Range:")
        print(f"Earliest: {train_tasks['deadline'].min()}")
        print(f"Latest: {train_tasks['deadline'].max()}")
        
        print("\nInitializing environment...")
        env = TodoListEnv(train_tasks, train_behavior, device=device)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(f"Environment initialized with state size: {state_size}, action size: {action_size}")
        
        # Validation environment
        val_env = None
        if test_tasks is not None:
            val_env = TodoListEnv(test_tasks, test_behavior, device=device)
        
        print("\nInitializing agent...")
        agent = DQNAgent(state_size, action_size, device=device)
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        validation_scores = []
        loss_history = []
        tasks_completed_per_episode = []
        
        best_validation_score = -float('inf')
        episodes_without_improvement = 0
        
        log_file = open("logs/training_log.txt", "w")
        
        print("\nStarting training loop...")
        for episode in range(episodes):
            # Reset environment and get initial state
            state, _ = env.reset()
            total_reward = 0
            episode_length = 0
            tasks_completed = 0
            
            # Run episode
            while True:
                # Get valid actions from environment
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                    
                # Agent selects action
                action = agent.act(state, valid_actions, training=True)
                
                # Take action in environment
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done or truncated)
                
                # Train agent
                loss = agent.replay(batch_size)
                if loss is not None:
                    loss_history.append(loss)
                
                state = next_state
                total_reward += reward
                episode_length += 1
                tasks_completed += 1
                
                if done or truncated:
                    break
            
            # Record episode metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            tasks_completed_per_episode.append(tasks_completed)
            
            # Calculate running average
            window_size = min(100, episode + 1)
            avg_reward = np.mean(episode_rewards[-window_size:])
            avg_length = np.mean(episode_lengths[-window_size:])
            
            # Get training stats
            stats = agent.get_training_stats()
            
            # Log progress
            log_message = (
                f"Episode: {episode+1}/{episodes}\n"
                f"Tasks Completed: {tasks_completed}, Total Reward: {total_reward:.3f}\n"
                f"Episode Length: {episode_length}, Avg Reward (last {window_size}): {avg_reward:.3f}\n"
                f"Avg Episode Length: {avg_length:.1f}\n"
                f"Epsilon: {stats['epsilon']:.3f}, Memory: {stats['memory_size']}\n"
                f"Avg Loss: {stats['avg_loss']:.6f}\n"
                f"------------------------\n"
            )
            
            if episode % 10 == 0 or episode < 10:
                print(log_message.strip())
            log_file.write(log_message)
            log_file.flush()  # Ensure logs are written
            
            # Validation
            if val_env is not None and (episode + 1) % validation_interval == 0:
                val_score = validate_agent(agent, val_env)
                validation_scores.append(val_score)
                print(f"Validation Score: {val_score:.3f}")
                log_file.write(f"Validation Score: {val_score:.3f}\n")
                
                # Early stopping
                if val_score > best_validation_score:
                    best_validation_score = val_score
                    episodes_without_improvement = 0
                    # Save best model
                    agent.save("models/best_model.pt")
                    print("New best model saved!")
                else:
                    episodes_without_improvement += validation_interval
                    
                if episodes_without_improvement >= early_stopping_patience:
                    print(f"Early stopping at episode {episode+1}")
                    break
            
            # Plotting
            if (episode + 1) % plot_interval == 0:
                create_training_plots(
                    episode_rewards, 
                    episode_lengths, 
                    loss_history, 
                    validation_scores, 
                    tasks_completed_per_episode,
                    episode + 1
                )
        
        # Final model save
        final_model_path = "models/dqn_model_final.pt"
        agent.save(final_model_path)
        print(f"\nTraining completed! Final model saved to {final_model_path}")
        
        log_file.close()
        
        # Final plots
        create_training_plots(
            episode_rewards, 
            episode_lengths, 
            loss_history, 
            validation_scores, 
            tasks_completed_per_episode,
            episodes,
            final=True
        )
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model(
        train_data_dir="data/train",
        test_data_dir="data/test",
        episodes=1000,
        batch_size=128,
        plot_interval=50,
        validation_interval=200,
        early_stopping_patience=500
    )