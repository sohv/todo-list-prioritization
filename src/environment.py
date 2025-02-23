import numpy as np
import pandas as pd
from gym import Env, spaces
from datetime import datetime

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
        self.action_space = spaces.Discrete(len(self.tasks))  # Actions: select a task
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.tasks) * 3,), dtype=np.float32)  # State: flattened task features
        
    def reset(self):
        self.current_task_index = 0
        return self._get_state()
    
    def step(self, action):
        # Execute the action (select a task)
        selected_task = self.tasks.iloc[action]
        
        # Calculate reward based on user behavior and task deadline
        reward = self._calculate_reward(selected_task)
        
        # Update the state
        self.current_task_index += 1
        next_state = self._get_state()
        
        # Check if all tasks are processed
        done = self.current_task_index >= len(self.tasks)
        
        return next_state, reward, done, {}
    
    def _get_state(self):
        # Return the current state (task features) as a flattened array
        state = self.tasks[['days_remaining', 'priority', 'estimated_time']].values
        return state.flatten()  # Flatten the 2D array to 1D
    
    def _calculate_reward(self, task):
        # Reward function based on task deadline and user behavior
        days_remaining = task['days_remaining']
        priority = task['priority']
        completion_time = self.user_behavior.loc[self.user_behavior['task_id'] == task['task_id'], 'completion_time'].values[0]
        
        # Reward is higher for completing high-priority tasks before their deadline
        if completion_time <= days_remaining:
            reward = priority * 10
        else:
            reward = -priority * 2 # Penalty for missing the deadline
        
        return reward