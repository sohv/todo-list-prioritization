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
        'status': [], 
        'start_date': [],  
        'completion_date': []  
    }
    
    if is_training:
        status_weights = [0.4, 0.3, 0.3]  # 40% completed, 30% in progress, 30% todo
    else:
        status_weights = [0.2, 0.3, 0.5]
        
    status_options = ['completed', 'in_progress', 'todo']
    
    for i in range(num_tasks):
        task_id = str(start_id + i)
        tasks['task_id'].append(task_id)
        
        # randomly assign status based on weights
        status = np.random.choice(status_options, p=status_weights)
        tasks['status'].append(status)
        
        # start date is within last 2 months for existing tasks
        start_date = current_date - timedelta(days=np.random.randint(0, 60))
        tasks['start_date'].append(start_date)
        
        if status == 'completed':
            deadline = start_date + timedelta(days=np.random.randint(1, 30))
            completion_date = deadline - timedelta(days=np.random.randint(0, 5))
            
        elif status == 'in_progress':
            # in-progress tasks have deadline within next 3 months
            deadline = current_date + timedelta(days=np.random.randint(1, 90))
            completion_date = None
            
        else:  # todo
            # todo tasks with deadline within next 5 months
            deadline = current_date + timedelta(days=np.random.randint(1, 150))
            completion_date = None
        
        tasks['deadline'].append(deadline)
        tasks['completion_date'].append(completion_date)

        days_to_deadline = (deadline - current_date).days
        if days_to_deadline < 30:
            priority_weights = [0.2, 0.3, 0.5] 
        else:
            priority_weights = [0.5, 0.3, 0.2] 
        
        tasks['priority'].append(np.random.choice([1, 2, 3], p=priority_weights))
        
        tasks['estimated_time'].append(round(np.random.uniform(0.5, 5.0), 6))
    
    tasks_df = pd.DataFrame(tasks)
    
    user_behavior = {
        'task_id': [],
        'completion_time': [],
        'actual_start_date': [],
        'actual_completion_date': []
    }
    
    for _, task in tasks_df.iterrows():
        if task['status'] in ['completed', 'in_progress']:
            user_behavior['task_id'].append(task['task_id'])
            
            estimated = task['estimated_time']
            variation = np.random.normal(0, 0.2)  # 20% variation
            completion_time = max(0.5, estimated * (1 + variation))
            user_behavior['completion_time'].append(round(completion_time, 6))
            
            user_behavior['actual_start_date'].append(task['start_date'])
            user_behavior['actual_completion_date'].append(task['completion_date'])
    
    user_behavior_df = pd.DataFrame(user_behavior)
    
    return tasks_df, user_behavior_df

def generate_train_test_data(train_size=70000, test_size=100):
    """Generate separate training and test datasets."""
    np.random.seed(42)
    
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    print(f"Generating training data ({train_size} tasks)...")
    train_tasks, train_behavior = generate_dataset(train_size, start_id=1, is_training=True)
    
    print(f"Generating test data ({test_size} tasks)...")
    test_tasks, test_behavior = generate_dataset(test_size, start_id=train_size+1, is_training=False)
    
    train_tasks.to_csv('data/train/tasks.csv', index=False)
    train_behavior.to_csv('data/train/user_behavior.csv', index=False)
    
    test_tasks.to_csv('data/test/tasks.csv', index=False)
    test_behavior.to_csv('data/test/user_behavior.csv', index=False)
    
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