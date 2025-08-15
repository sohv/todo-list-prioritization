import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import Env, spaces
from datetime import datetime

from reward_function import calculate_reward

class TodoListEnv(Env):
    def __init__(self, tasks, user_behavior, device=None):
        super(TodoListEnv, self).__init__()
        self.tasks = tasks
        self.user_behavior = user_behavior
        self.current_task_index = 0
        self.device = device if device is not None else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tasks['deadline'] = pd.to_datetime(self.tasks['deadline'])
        self.tasks['days_remaining'] = (self.tasks['deadline'] - datetime.now()).dt.days
        self.action_space = spaces.Discrete(len(self.tasks))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.tasks) * 3,), dtype=np.float32)

    def reset(self):
        self.current_task_index = 0
        return self._get_state()
    
    def step(self, action):
        selected_task = self.tasks.iloc[action]
        reward = calculate_reward(selected_task, self.user_behavior, self.current_task_index)
        self.current_task_index += 1
        next_state = self._get_state()
        done = self.current_task_index >= len(self.tasks)
        return next_state, reward, done, {}
    
    def _get_state(self):
        state = self.tasks[['days_remaining', 'priority', 'estimated_time']].values
        return state.flatten()