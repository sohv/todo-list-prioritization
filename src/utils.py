# utility functions (data preprocessing, reward function, etc.)

import pandas as pd
import numpy as np
import logging
import os

def load_data():
    tasks = pd.read_csv("data/tasks.csv")
    user_behavior = pd.read_csv("data/user_behavior.csv")
    return tasks, user_behavior

def setup_logging(log_dir='logs', log_file='training.log', level=logging.INFO):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_model(agent, env, num_episodes=5):
    """Validate trained model on validation environment"""
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions() if hasattr(env, 'get_valid_actions') else None
            
            # Create task mask
            task_mask = None
            if valid_actions is not None:
                import torch
                task_mask = torch.zeros(env.max_tasks_per_episode)
                task_mask[valid_actions] = 1.0
            
            # Select action (deterministic for validation)
            action, _ = agent.act(state, task_mask, deterministic=True)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            state = next_state
            episode_reward += reward
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)