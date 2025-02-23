import pandas as pd
import numpy as np

# Generate task data
tasks = pd.DataFrame({
    'task_id': range(1, 101),
    'task_name': [f'Task {i}' for i in range(1, 101)],
    'deadline': pd.date_range(start='2023-11-25', periods=100, freq='D'),
    'priority': np.random.choice([1, 2, 3], size=100),
    'estimated_time': np.random.uniform(0.5, 5.0, size=100),
    'category': np.random.choice(['work', 'personal', 'health'], size=100)
})

# Generate user behavior data
user_behavior = pd.DataFrame({
    'task_id': range(1, 101),
    'completion_time': np.random.uniform(0.5, 5.0, size=100),
    'completion_status': np.random.choice([0, 1], size=100),
    'user_rating': np.random.choice([1, 2, 3, 4, 5], size=100)
})

# Save to CSV
tasks.to_csv('data/tasks.csv', index=False)
user_behavior.to_csv('data/user_behavior.csv', index=False)