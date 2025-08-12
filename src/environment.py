import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import Env, spaces
from datetime import datetime

from .reward_function import calculate_reward

class TodoListEnv(Env):
    def __init__(self, tasks, user_behavior, device=None):
        super(TodoListEnv, self).__init__()
        self.original_tasks = tasks.copy()
        self.user_behavior = user_behavior
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Process initial tasks
        self.original_tasks['deadline'] = pd.to_datetime(self.original_tasks['deadline'])
        self.original_tasks['start_date'] = pd.to_datetime(self.original_tasks['start_date'])
        
        # Dynamic state tracking
        self.available_tasks = None
        self.completed_tasks = []
        self.episode_step = 0
        
        # Fixed action and observation spaces based on maximum possible tasks
        self.max_tasks = len(self.original_tasks)
        self.action_space = spaces.Discrete(self.max_tasks)
        # State: [task_features..., action_mask]
        self.observation_space = spaces.Box(
            low=-10, high=10, 
            shape=(self.max_tasks * 4 + self.max_tasks,), 
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Reset available tasks (only incomplete tasks)
        self.available_tasks = self.original_tasks.copy()
        self.completed_tasks = []
        self.episode_step = 0
        
        # Update days remaining based on current date
        current_date = datetime.now()
        self.available_tasks['days_remaining'] = (
            self.available_tasks['deadline'] - current_date
        ).dt.days
        
        return self._get_state(), {}
    
    def step(self, action):
        if action >= len(self.available_tasks):
            # Invalid action - heavily penalize
            return self._get_state(), -1.0, True, False, {"invalid_action": True}
        
        # Get selected task
        selected_task = self.available_tasks.iloc[action].copy()
        
        # Calculate reward
        reward = calculate_reward(
            selected_task, 
            self.user_behavior, 
            self.episode_step,
            self.available_tasks
        )
        
        # Mark task as completed and remove from available tasks
        self.completed_tasks.append(selected_task)
        self.available_tasks = self.available_tasks.drop(self.available_tasks.index[action]).reset_index(drop=True)
        
        self.episode_step += 1
        
        # Episode ends when no tasks left or max steps reached
        done = len(self.available_tasks) == 0 or self.episode_step >= self.max_tasks
        truncated = self.episode_step >= self.max_tasks and len(self.available_tasks) > 0
        
        return self._get_state(), reward, done, truncated, {}
    
    def _get_state(self):
        # Create normalized state representation
        state_features = []
        action_mask = []
        
        current_date = datetime.now()
        
        for i in range(self.max_tasks):
            if i < len(self.available_tasks):
                task = self.available_tasks.iloc[i]
                
                # Normalize features
                days_remaining = (task['deadline'] - current_date).days
                normalized_days = np.clip(days_remaining / 30.0, -3, 3)  # Normalize to ~[-3,3]
                normalized_priority = (task['priority'] - 2) / 1.0  # Assuming priority 1-3, normalize to [-1,1]
                normalized_time = np.clip((task['estimated_time'] - 2.5) / 2.5, -1, 3)  # Normalize estimated time
                
                # Task urgency (inverse of days remaining, clamped)
                urgency = np.clip(1.0 / max(abs(days_remaining), 0.1), 0, 10) if days_remaining > 0 else 10
                urgency = np.clip((urgency - 2) / 2, -1, 3)
                
                state_features.extend([normalized_days, normalized_priority, normalized_time, urgency])
                action_mask.append(1.0)  # Available action
            else:
                # Padding for unavailable task slots
                state_features.extend([0.0, 0.0, 0.0, 0.0])
                action_mask.append(0.0)  # Unavailable action
        
        return np.array(state_features + action_mask, dtype=np.float32)
    
    def get_valid_actions(self):
        """Get list of valid action indices"""
        return list(range(len(self.available_tasks)))