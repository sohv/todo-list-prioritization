import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
from datetime import datetime
from .environment import TodoListEnv
from .a2c_agent import A2CAgent
from .a3c_agent import A3CAgent

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_directories():
    """Create necessary directories for saving models and plots"""
    directories = ['models', 'plots', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_env_factory(train_tasks, train_behavior):
    """Create environment factory for A3C workers"""
    def env_factory():
        return TodoListEnv(train_tasks, train_behavior, device=torch.device("cpu"))
    return env_factory

def train_a2c_model(
    train_data_dir="data/train",
    test_data_dir="data/test",
    episodes=1000,
    max_steps_per_episode=500,
    update_frequency=20,
    save_interval=100,
    plot_interval=50,
    validation_interval=100,
    early_stopping_patience=200
):
    """Train A2C agent on task prioritization"""
    print("=== Training A2C Agent ===")
    create_directories()
    
    try:
        # Load data
        print("Loading training data...")
        train_tasks = pd.read_csv(os.path.join(train_data_dir, 'tasks.csv'))
        train_behavior = pd.read_csv(os.path.join(train_data_dir, 'user_behavior.csv'))
        print(f"Loaded {len(train_tasks)} training tasks")
        
        # Load test data if available
        test_env = None
        if os.path.exists(test_data_dir):
            try:
                test_tasks = pd.read_csv(os.path.join(test_data_dir, 'tasks.csv'))
                test_behavior = pd.read_csv(os.path.join(test_data_dir, 'user_behavior.csv'))
                test_env = TodoListEnv(test_tasks, test_behavior, device=device)
                print(f"Loaded {len(test_tasks)} test tasks for validation")
            except:
                print("Test data not found, skipping validation")
        
        # Initialize environment and agent
        print("Initializing environment and A2C agent...")
        env = TodoListEnv(train_tasks, train_behavior, device=device)
        state_size = env.observation_space.shape[0]
        max_tasks = env.action_space.n
        
        agent = A2CAgent(
            state_dim=state_size,
            max_tasks=max_tasks,
            device=device,
            lr_actor=3e-4,
            lr_critic=1e-3
        )
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        validation_scores = []
        actor_losses = []
        critic_losses = []
        entropies = []
        
        best_validation_score = -float('inf')
        episodes_without_improvement = 0
        
        log_file = open("logs/a2c_training_log.txt", "w")
        
        print("Starting A2C training loop...")
        start_time = time.time()
        
        for episode in range(episodes):
            # Reset environment
            state, _ = env.reset()
            total_reward = 0
            episode_length = 0
            
            # Run episode
            while True:
                # Get valid actions
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                
                # Create task mask
                task_mask = torch.zeros(max_tasks)
                task_mask[valid_actions] = 1.0
                
                # Agent selects action
                action, log_prob, info = agent.act(state, task_mask.numpy(), deterministic=False)
                
                # Take action
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done or truncated, log_prob)
                
                state = next_state
                total_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
                
                # Update networks periodically
                if len(agent.states) >= update_frequency:
                    update_stats = agent.update()
                    if update_stats:
                        actor_losses.append(update_stats['actor_loss'])
                        critic_losses.append(update_stats['critic_loss'])
                        entropies.append(update_stats['entropy'])
            
            # Final update at episode end
            if len(agent.states) > 0:
                update_stats = agent.update()
                if update_stats:
                    actor_losses.append(update_stats['actor_loss'])
                    critic_losses.append(update_stats['critic_loss'])
                    entropies.append(update_stats['entropy'])
            
            # Record metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            
            # Calculate running averages
            window_size = min(100, episode + 1)
            avg_reward = np.mean(episode_rewards[-window_size:])
            avg_length = np.mean(episode_lengths[-window_size:])
            
            # Log progress
            if episode % 10 == 0 or episode < 10:
                elapsed_time = time.time() - start_time
                log_message = (
                    f"Episode {episode+1}/{episodes} - "
                    f"Reward: {total_reward:.3f}, Length: {episode_length}, "
                    f"Avg Reward: {avg_reward:.3f}, Avg Length: {avg_length:.1f}, "
                    f"Time: {elapsed_time:.1f}s"
                )
                
                if actor_losses:
                    log_message += f", Actor Loss: {np.mean(actor_losses[-10:]):.6f}"
                if critic_losses:
                    log_message += f", Critic Loss: {np.mean(critic_losses[-10:]):.6f}"
                if entropies:
                    log_message += f", Entropy: {np.mean(entropies[-10:]):.3f}"
                
                print(log_message)
                log_file.write(log_message + "\n")
                log_file.flush()
            
            # Validation
            if test_env is not None and (episode + 1) % validation_interval == 0:
                val_score = validate_agent(agent, test_env, num_episodes=5)
                validation_scores.append(val_score)
                print(f"Validation Score: {val_score:.3f}")
                log_file.write(f"Validation Score: {val_score:.3f}\n")
                
                # Early stopping
                if val_score > best_validation_score:
                    best_validation_score = val_score
                    episodes_without_improvement = 0
                    agent.save("models/best_a2c_model.pt")
                    print("New best A2C model saved!")
                else:
                    episodes_without_improvement += validation_interval
                
                if episodes_without_improvement >= early_stopping_patience:
                    print(f"Early stopping at episode {episode+1}")
                    break
            
            # Save checkpoints
            if (episode + 1) % save_interval == 0:
                model_path = f"models/a2c_model_episode_{episode+1}.pt"
                agent.save(model_path)
                print(f"A2C model checkpoint saved to {model_path}")
            
            # Create plots
            if (episode + 1) % plot_interval == 0:
                create_a2c_plots(
                    episode_rewards, episode_lengths, actor_losses, 
                    critic_losses, entropies, validation_scores, episode + 1
                )
        
        # Final save
        agent.save("models/a2c_model_final.pt")
        log_file.close()
        
        # Final plots
        create_a2c_plots(
            episode_rewards, episode_lengths, actor_losses,
            critic_losses, entropies, validation_scores, episodes, final=True
        )
        
        print(f"A2C training completed! Final average reward: {np.mean(episode_rewards[-100:]):.3f}")
        
        return agent
        
    except Exception as e:
        print(f"Error during A2C training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def train_a3c_model(
    train_data_dir="data/train",
    test_data_dir="data/test",
    num_workers=4,
    episodes_per_worker=500,
    max_steps_per_episode=500,
    update_frequency=20
):
    """Train A3C agent on task prioritization"""
    print("=== Training A3C Agent ===")
    create_directories()
    
    try:
        # Load data
        print("Loading training data...")
        train_tasks = pd.read_csv(os.path.join(train_data_dir, 'tasks.csv'))
        train_behavior = pd.read_csv(os.path.join(train_data_dir, 'user_behavior.csv'))
        print(f"Loaded {len(train_tasks)} training tasks")
        
        # Initialize environment to get dimensions
        env = TodoListEnv(train_tasks, train_behavior, device=device)
        state_size = env.observation_space.shape[0]
        max_tasks = env.action_space.n
        
        # Create environment factory
        env_factory = create_env_factory(train_tasks, train_behavior)
        
        # Initialize A3C agent
        print(f"Initializing A3C agent with {num_workers} workers...")
        agent = A3CAgent(
            state_dim=state_size,
            max_tasks=max_tasks,
            num_workers=num_workers,
            device=device
        )
        
        # Train A3C
        start_time = time.time()
        agent.train(
            env_factory=env_factory,
            num_episodes_per_worker=episodes_per_worker,
            max_steps_per_episode=max_steps_per_episode,
            update_frequency=update_frequency
        )
        
        training_time = time.time() - start_time
        print(f"A3C training completed in {training_time:.1f} seconds")
        
        # Save model
        agent.save("models/a3c_model_final.pt")
        
        # Create plots
        if agent.global_episode_rewards:
            create_a3c_plots(agent)
        
        print(f"A3C training completed! Final stats: {agent.get_training_stats()}")
        
        return agent
        
    except Exception as e:
        print(f"Error during A3C training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def validate_agent(agent, val_env, num_episodes=10):
    """Validate agent performance"""
    total_scores = []
    
    for _ in range(num_episodes):
        state, _ = val_env.reset()
        total_reward = 0
        
        while True:
            valid_actions = val_env.get_valid_actions()
            if not valid_actions:
                break
            
            # Create task mask
            task_mask = torch.zeros(agent.max_tasks if hasattr(agent, 'max_tasks') else len(valid_actions))
            task_mask[valid_actions] = 1.0
            
            # Get action (deterministic for evaluation)
            if hasattr(agent, 'act'):
                if isinstance(agent, A2CAgent):
                    action, _, info = agent.act(state, task_mask.numpy(), deterministic=True)
                else:  # A3C
                    action, info = agent.act(state, task_mask.numpy(), deterministic=True)
            else:
                action = np.random.choice(valid_actions)
            
            state, reward, done, truncated, _ = val_env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        total_scores.append(total_reward)
    
    return np.mean(total_scores)

def create_a2c_plots(episode_rewards, episode_lengths, actor_losses, critic_losses, 
                     entropies, validation_scores, episode_num, final=False):
    """Create A2C training plots"""
    suffix = "final" if final else f"episode_{episode_num}"
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("A2C Training Progress", fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    if len(episode_rewards) > 100:
        running_avg = [np.mean(episode_rewards[max(0, i-99):i+1]) for i in range(len(episode_rewards))]
        axes[0, 0].plot(running_avg, 'r-', alpha=0.7, label='Running avg (100)')
        axes[0, 0].legend()
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Actor loss
    if actor_losses:
        axes[0, 2].plot(actor_losses)
        axes[0, 2].set_title('Actor Loss')
        axes[0, 2].set_xlabel('Update Step')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True)
    else:
        axes[0, 2].text(0.5, 0.5, 'No Actor Loss Data', ha='center', va='center')
        axes[0, 2].set_title('Actor Loss')
    
    # Critic loss
    if critic_losses:
        axes[1, 0].plot(critic_losses)
        axes[1, 0].set_title('Critic Loss')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Critic Loss Data', ha='center', va='center')
        axes[1, 0].set_title('Critic Loss')
    
    # Entropy
    if entropies:
        axes[1, 1].plot(entropies)
        axes[1, 1].set_title('Policy Entropy')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Entropy Data', ha='center', va='center')
        axes[1, 1].set_title('Policy Entropy')
    
    # Validation scores
    if validation_scores:
        val_episodes = np.arange(len(validation_scores)) * 100  # Assuming validation every 100 episodes
        axes[1, 2].plot(val_episodes, validation_scores, 'ro-')
        axes[1, 2].set_title('Validation Performance')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Validation Score')
        axes[1, 2].grid(True)
    else:
        axes[1, 2].text(0.5, 0.5, 'No Validation Data', ha='center', va='center')
        axes[1, 2].set_title('Validation Performance')
    
    plt.tight_layout()
    plt.savefig(f'plots/a2c_training_progress_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_a3c_plots(agent):
    """Create A3C training plots"""
    if not agent.global_episode_rewards:
        print("No A3C training data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("A3C Training Results", fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(agent.global_episode_rewards)
    if len(agent.global_episode_rewards) > 100:
        running_avg = [np.mean(list(agent.global_episode_rewards)[max(0, i-99):i+1]) 
                      for i in range(len(agent.global_episode_rewards))]
        axes[0, 0].plot(running_avg, 'r-', alpha=0.7, label='Running avg (100)')
        axes[0, 0].legend()
    axes[0, 0].set_title('Episode Rewards (All Workers)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(agent.global_episode_lengths)
    axes[0, 1].set_title('Episode Lengths (All Workers)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Reward distribution
    axes[1, 0].hist(agent.global_episode_rewards, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Total Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True)
    axes[1, 0].axvline(np.mean(agent.global_episode_rewards), color='red', 
                      linestyle='--', label=f'Mean: {np.mean(agent.global_episode_rewards):.3f}')
    axes[1, 0].legend()
    
    # Learning curve (smoothed)
    if len(agent.global_episode_rewards) > 50:
        window = min(50, len(agent.global_episode_rewards) // 10)
        smoothed = [np.mean(list(agent.global_episode_rewards)[max(0, i-window):i+1]) 
                   for i in range(len(agent.global_episode_rewards))]
        axes[1, 1].plot(smoothed)
        axes[1, 1].set_title(f'Learning Curve (smoothed, window={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Smoothed Reward')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data for learning curve', 
                       ha='center', va='center')
        axes[1, 1].set_title('Learning Curve')
    
    plt.tight_layout()
    plt.savefig('plots/a3c_training_results_final.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_algorithms(a2c_agent=None, a3c_agent=None, test_data_dir="data/test", num_eval_episodes=20):
    """Compare A2C and A3C performance"""
    print("=== Algorithm Comparison ===")
    
    if not os.path.exists(test_data_dir):
        print("Test data not found, skipping comparison")
        return
    
    try:
        # Load test data
        test_tasks = pd.read_csv(os.path.join(test_data_dir, 'tasks.csv'))
        test_behavior = pd.read_csv(os.path.join(test_data_dir, 'user_behavior.csv'))
        test_env = TodoListEnv(test_tasks, test_behavior, device=device)
        
        results = {}
        
        # Test A2C
        if a2c_agent is not None:
            print("Evaluating A2C agent...")
            a2c_score = validate_agent(a2c_agent, test_env, num_eval_episodes)
            results['A2C'] = a2c_score
            print(f"A2C average score: {a2c_score:.3f}")
        
        # Test A3C
        if a3c_agent is not None:
            print("Evaluating A3C agent...")
            a3c_score = validate_agent(a3c_agent, test_env, num_eval_episodes)
            results['A3C'] = a3c_score
            print(f"A3C average score: {a3c_score:.3f}")
        
        # Create comparison plot
        if results:
            plt.figure(figsize=(10, 6))
            algorithms = list(results.keys())
            scores = list(results.values())
            
            bars = plt.bar(algorithms, scores, alpha=0.7, color=['blue', 'orange'])
            plt.title('Algorithm Performance Comparison')
            plt.ylabel('Average Test Score')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('plots/algorithm_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Comparison plot saved to plots/algorithm_comparison.png")
        
        return results
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        return {}

if __name__ == "__main__":
    # Train both algorithms
    print("Training Actor-Critic algorithms for task prioritization...\n")
    
    # Train A2C
    a2c_agent = train_a2c_model(
        episodes=500,  # Reduced for testing
        max_steps_per_episode=200,
        update_frequency=10,
        save_interval=50,
        plot_interval=25
    )
    
    print("\n" + "="*60 + "\n")
    
    # Train A3C
    a3c_agent = train_a3c_model(
        num_workers=2,  # Reduced for testing
        episodes_per_worker=250,
        max_steps_per_episode=200,
        update_frequency=10
    )
    
    print("\n" + "="*60 + "\n")
    
    # Compare algorithms
    comparison_results = compare_algorithms(a2c_agent, a3c_agent)
    
    print("\n=== Training Complete ===")
    if comparison_results:
        print("Final Comparison Results:")
        for alg, score in comparison_results.items():
            print(f"  {alg}: {score:.3f}")
    
    print("\nModels and plots saved in respective directories.")
    print("Training logs available in logs/ directory.")