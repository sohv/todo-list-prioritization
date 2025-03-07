import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_dataset(num_tasks, start_id=1, is_training=True):
    """
    Generate dataset with task status and realistic timeframes.
    """
    current_date = datetime.now()
    
    tasks = {
        'task_id': [],
        'deadline': [],
        'priority': [],
        'estimated_time': [],
        'status': [],  # New field: 'completed', 'in_progress', 'todo'
        'start_date': [],  # New field: when the task was created/started
        'completion_date': []  # New field: when the task was completed (if applicable)
    }
    
    # Status distribution for training set
    if is_training:
        status_weights = [0.4, 0.3, 0.3]  # 40% completed, 30% in progress, 30% todo
    else:
        status_weights = [0.2, 0.3, 0.5]  # Test set: more todo tasks
        
    status_options = ['completed', 'in_progress', 'todo']
    
    for i in range(num_tasks):
        task_id = str(start_id + i)
        tasks['task_id'].append(task_id)
        
        # Randomly assign status based on weights
        status = np.random.choice(status_options, p=status_weights)
        tasks['status'].append(status)
        
        # Start date: within last 2 months for existing tasks
        start_date = current_date - timedelta(days=np.random.randint(0, 60))
        tasks['start_date'].append(start_date)
        
        # Deadline and completion dates based on status
        if status == 'completed':
            # Completed tasks: deadline and completion date in the past
            deadline = start_date + timedelta(days=np.random.randint(1, 30))
            completion_date = deadline - timedelta(days=np.random.randint(0, 5))
            
        elif status == 'in_progress':
            # In-progress tasks: deadline within next 3 months
            deadline = current_date + timedelta(days=np.random.randint(1, 90))
            completion_date = None
            
        else:  # todo
            # Todo tasks: deadline within next 5 months
            deadline = current_date + timedelta(days=np.random.randint(1, 150))
            completion_date = None
        
        tasks['deadline'].append(deadline)
        tasks['completion_date'].append(completion_date)
        
        # Priority (1-3 to match original)
        # Higher priority for closer deadlines
        days_to_deadline = (deadline - current_date).days
        if days_to_deadline < 30:
            priority_weights = [0.2, 0.3, 0.5]  # More likely to be high priority
        else:
            priority_weights = [0.5, 0.3, 0.2]  # More likely to be low priority
        
        tasks['priority'].append(np.random.choice([1, 2, 3], p=priority_weights))
        
        # Estimated time (0.5 to 5.0 hours)
        tasks['estimated_time'].append(round(np.random.uniform(0.5, 5.0), 6))
    
    tasks_df = pd.DataFrame(tasks)
    
    # Generate user behavior data
    user_behavior = {
        'task_id': [],
        'completion_time': [],
        'actual_start_date': [],
        'actual_completion_date': []
    }
    
    for _, task in tasks_df.iterrows():
        if task['status'] in ['completed', 'in_progress']:
            user_behavior['task_id'].append(task['task_id'])
            
            # Completion time based on estimated time
            estimated = task['estimated_time']
            variation = np.random.normal(0, 0.2)  # 20% variation
            completion_time = max(0.5, estimated * (1 + variation))
            user_behavior['completion_time'].append(round(completion_time, 6))
            
            # Actual dates for completed/in-progress tasks
            user_behavior['actual_start_date'].append(task['start_date'])
            user_behavior['actual_completion_date'].append(task['completion_date'])
    
    user_behavior_df = pd.DataFrame(user_behavior)
    
    return tasks_df, user_behavior_df

def generate_train_test_data(train_size=70000, test_size=100):
    """Generate separate training and test datasets."""
    np.random.seed(42)
    
    # Create data directories
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    # Generate training data
    print(f"Generating training data ({train_size} tasks)...")
    train_tasks, train_behavior = generate_dataset(train_size, start_id=1, is_training=True)
    
    # Generate test data
    print(f"Generating test data ({test_size} tasks)...")
    test_tasks, test_behavior = generate_dataset(test_size, start_id=train_size+1, is_training=False)
    
    # Save training data
    train_tasks.to_csv('data/train/tasks.csv', index=False)
    train_behavior.to_csv('data/train/user_behavior.csv', index=False)
    
    # Save test data
    test_tasks.to_csv('data/test/tasks.csv', index=False)
    test_behavior.to_csv('data/test/user_behavior.csv', index=False)
    
    # Print summary statistics
    print("\nTraining Data Summary:")
    print("\nTask Status Distribution:")
    print(train_tasks['status'].value_counts(normalize=True))
    print("\nPriority Distribution:")
    print(train_tasks['priority'].value_counts(normalize=True))
    print("\nDeadline Range:")
    print(f"Earliest: {train_tasks['deadline'].min()}")
    print(f"Latest: {train_tasks['deadline'].max()}")
    
    print("\nTest Data Summary:")
    print("\nTask Status Distribution:")
    print(test_tasks['status'].value_counts(normalize=True))
    print("\nPriority Distribution:")
    print(test_tasks['priority'].value_counts(normalize=True))
    print("\nDeadline Range:")
    print(f"Earliest: {test_tasks['deadline'].min()}")
    print(f"Latest: {test_tasks['deadline'].max()}")

if __name__ == "__main__":
    generate_train_test_data()
    print("\nData files created in data/train and data/test directories")