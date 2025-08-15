import sys
sys.path.append('src')
from reward_function import calculate_reward
from simple_reward_function import calculate_simple_reward
from datetime import datetime, timedelta
import pandas as pd

# Test scenarios
scenarios = [
    {'name': 'High priority, urgent', 'priority': 3, 'days': 1},
    {'name': 'Medium priority, normal', 'priority': 2, 'days': 7},
    {'name': 'Low priority, distant', 'priority': 1, 'days': 30},
    {'name': 'Overdue task', 'priority': 2, 'days': -2},
]

user_behavior = pd.DataFrame()
print('Scenario                | Complex | Simple | Diff')
print('------------------------|---------|--------|--------')

for scenario in scenarios:
    task = {
        'task_id': f'test_{scenario["name"]}',
        'priority': scenario['priority'],
        'deadline': datetime.now() + timedelta(days=scenario['days']),
        'estimated_time': 2.0,
        'type': 'work',
        'status': 'pending'
    }
    
    complex_reward = calculate_reward(task, user_behavior, 0, None)
    simple_reward = calculate_simple_reward(task, user_behavior, 0)
    diff = simple_reward - complex_reward
    
    print(f'{scenario["name"]:22} | {complex_reward:7.3f} | {simple_reward:6.3f} | {diff:6.3f}')
