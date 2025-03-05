import numpy as np
import pandas as pd
from gym import Env, spaces
from datetime import datetime

# Import reward function from external file
from src.reward_function import calculate_reward

class TodoListEnv(Env):
    def __init__(self, tasks, user_behavior):
        super(TodoListEnv, self).__init__()
        self.tasks = tasks
        self.user_behavior = user_behavior
        self.current_task_index = 0
        
        # Convert deadline to datetime and calculate days remaining
        self.tasks['deadline'] = pd.to_datetime(self.tasks['deadline'])
        self.tasks['days_remaining'] = (self.tasks['deadline'] - datetime.now()).dt.days
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.tasks))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.tasks) * 3,), dtype=np.float32)

    def reset(self):
        self.current_task_index = 0
        return self._get_state()
    
    def step(self, action):
        selected_task = self.tasks.iloc[action]
        
        # Use the external reward function
        reward = calculate_reward(selected_task, self.user_behavior, self.current_task_index)
        
        self.current_task_index += 1
        next_state = self._get_state()
        
        done = self.current_task_index >= len(self.tasks)
        
        return next_state, reward, done, {}
    
    def _get_state(self):
        state = self.tasks[['days_remaining', 'priority', 'estimated_time']].values
        return state.flatten()