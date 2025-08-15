#!/usr/bin/env python3
"""
Simple RL Evaluation: A2C vs DQN
Focus on actual RL metrics that matter
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.append('/root/todo')
from src.environment import TodoListEnv
from src.a2c_agent import A2CAgent
from src.dqn_agent import DQNAgent

def evaluate_agent(agent, test_data_dir="data/test", num_episodes=10):
    """Simple RL evaluation - just run episodes and measure performance"""
    
    # Load test data
    tasks_df = pd.read_csv(f"{test_data_dir}/tasks.csv")
    user_behavior_df = pd.read_csv(f"{test_data_dir}/user_behavior.csv")
    
    # Get max_tasks from loaded model if possible
    max_tasks = 300  # default
    if hasattr(agent, 'max_tasks'):
        max_tasks = agent.max_tasks
    elif hasattr(agent, 'action_size'):
        max_tasks = agent.action_size
    
    # Create environment
    env = TodoListEnv(tasks_df, user_behavior_df, max_tasks_per_episode=max_tasks)
    
    episode_rewards = []
    episode_lengths = []
    completion_rates = []
    
    print(f"Running {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        completed_tasks = 0
        
        for step in range(100):  # Max 100 steps per episode
            # Get action from agent
            if hasattr(agent, 'actor'):  # A2C
                action, _, _ = agent.act(state, deterministic=True)
            else:  # DQN
                action = agent.act(state, training=False)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if reward > 0:  # Task completed
                completed_tasks += 1
            
            state = next_state
            
            if done or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        completion_rates.append(completed_tasks / steps if steps > 0 else 0)
        
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, Steps={steps}, Completed={completed_tasks}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_completion_rate': np.mean(completion_rates),
        'episode_rewards': episode_rewards
    }

def compare_agents():
    """Simple comparison between A2C and DQN"""
    
    print("=== RL ALGORITHM COMPARISON ===\n")
    
    # Load A2C
    print("Loading A2C agent...")
    a2c_path = "models/a2c_model_final.pt"
    if os.path.exists(a2c_path):
        # Get dimensions from saved model
        checkpoint = torch.load(a2c_path, map_location='cpu', weights_only=False)
        if 'hyperparameters' in checkpoint:
            state_dim = checkpoint['hyperparameters']['state_dim']
            max_tasks = checkpoint['hyperparameters']['max_tasks']
        else:
            state_dim, max_tasks = 500, 100  # fallback
        
        a2c_agent = A2CAgent(state_dim=state_dim, max_tasks=max_tasks)
        a2c_agent.load(a2c_path)
        
        print("Evaluating A2C...")
        a2c_results = evaluate_agent(a2c_agent)
    else:
        print("A2C model not found!")
        a2c_results = None
    
    # Load DQN
    print("\nLoading DQN agent...")
    dqn_path = "models/dqn_model_final.pt"
    if os.path.exists(dqn_path):
        # Get dimensions from saved model
        checkpoint = torch.load(dqn_path, map_location='cpu', weights_only=False)
        state_size = checkpoint.get('state_size', 500)
        action_size = checkpoint.get('action_size', 100)
        
        dqn_agent = DQNAgent(state_size=state_size, action_size=action_size)
        dqn_agent.load(dqn_path)
        
        print("Evaluating DQN...")
        dqn_results = evaluate_agent(dqn_agent)
    else:
        print("DQN model not found!")
        dqn_results = None
    
    # Compare results
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    
    if a2c_results:
        print(f"\nA2C Results:")
        print(f"  Mean Reward: {a2c_results['mean_reward']:.3f} ± {a2c_results['std_reward']:.3f}")
        print(f"  Mean Episode Length: {a2c_results['mean_length']:.1f}")
        print(f"  Mean Completion Rate: {a2c_results['mean_completion_rate']:.3f}")
    
    if dqn_results:
        print(f"\nDQN Results:")
        print(f"  Mean Reward: {dqn_results['mean_reward']:.3f} ± {dqn_results['std_reward']:.3f}")
        print(f"  Mean Episode Length: {dqn_results['mean_length']:.1f}")
        print(f"  Mean Completion Rate: {dqn_results['mean_completion_rate']:.3f}")
    
    # Determine winner
    if a2c_results and dqn_results:
        print(f"\n🏆 WINNER:")
        if a2c_results['mean_reward'] > dqn_results['mean_reward']:
            diff = a2c_results['mean_reward'] - dqn_results['mean_reward']
            print(f"  A2C wins by {diff:.3f} reward points!")
        elif dqn_results['mean_reward'] > a2c_results['mean_reward']:
            diff = dqn_results['mean_reward'] - a2c_results['mean_reward']
            print(f"  DQN wins by {diff:.3f} reward points!")
        else:
            print(f"  It's a tie!")
    
    # Simple plot
    if a2c_results and dqn_results:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.bar(['A2C', 'DQN'], [a2c_results['mean_reward'], dqn_results['mean_reward']])
        plt.title('Mean Reward')
        plt.ylabel('Reward')
        
        plt.subplot(1, 3, 2)
        plt.bar(['A2C', 'DQN'], [a2c_results['mean_length'], dqn_results['mean_length']])
        plt.title('Mean Episode Length')
        plt.ylabel('Steps')
        
        plt.subplot(1, 3, 3)
        plt.bar(['A2C', 'DQN'], [a2c_results['mean_completion_rate'], dqn_results['mean_completion_rate']])
        plt.title('Mean Completion Rate')
        plt.ylabel('Rate')
        
        plt.tight_layout()
        plt.savefig('results/simple_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: results/simple_comparison.png")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    compare_agents()
