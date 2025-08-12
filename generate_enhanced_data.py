#!/usr/bin/env python3
"""
Enhanced Dataset Generator for RL Training

Addresses critical limitations in the original dataset:
1. Fresh, realistic temporal data
2. Rich task features and context
3. Multi-user scenarios  
4. Complex task relationships
5. Comprehensive behavioral patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import random
import string

def generate_fake_name():
    first_names = ['Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry', 'Iris', 'Jack']
    last_names = ['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson', 'Miller', 'Jones', 'Garcia', 'Martinez', 'Anderson']
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_fake_text(max_chars=200):
    words = ['task', 'project', 'implement', 'review', 'update', 'fix', 'create', 'develop', 'test', 
            'optimize', 'refactor', 'design', 'analyze', 'integrate', 'deploy', 'maintain', 'document',
            'feature', 'bug', 'enhancement', 'system', 'database', 'interface', 'component', 'service']
    
    text = ' '.join(random.choices(words, k=random.randint(10, 30)))
    return text[:max_chars] if len(text) > max_chars else text

def generate_fake_phrase():
    adjectives = ['innovative', 'robust', 'scalable', 'efficient', 'secure', 'user-friendly', 'reliable']
    nouns = ['solution', 'platform', 'system', 'framework', 'application', 'service', 'tool']
    return f"{random.choice(adjectives)} {random.choice(nouns)}"

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Enhanced task types with realistic distributions
TASK_TYPES = {
    'coding': 0.25,
    'review': 0.15, 
    'documentation': 0.12,
    'meeting': 0.10,
    'testing': 0.10,
    'research': 0.08,
    'planning': 0.07,
    'bug_fix': 0.06,
    'deployment': 0.04,
    'training': 0.03
}

# Project contexts for realistic task relationships
PROJECTS = ['web_app', 'mobile_app', 'api_service', 'data_pipeline', 'ml_model']

# User personas with different working patterns
USER_PERSONAS = {
    'efficiency_focused': {'task_switching_penalty': 0.8, 'prefers_batch': True, 'work_hours': (9, 17)},
    'flexible_worker': {'task_switching_penalty': 0.3, 'prefers_batch': False, 'work_hours': (8, 20)},
    'deadline_driven': {'task_switching_penalty': 0.5, 'prefers_batch': True, 'work_hours': (7, 19)},
    'collaborative': {'task_switching_penalty': 0.4, 'prefers_batch': False, 'work_hours': (9, 18)}
}

def generate_enhanced_dataset(num_tasks=10000, num_users=5, lookback_days=90, future_days=180):
    """
    Generate comprehensive dataset with realistic task relationships and user behaviors.
    """
    current_date = datetime.now()
    start_date = current_date - timedelta(days=lookback_days)
    end_date = current_date + timedelta(days=future_days)
    
    # Generate users with personas
    users = []
    for i in range(num_users):
        persona = np.random.choice(list(USER_PERSONAS.keys()))
        users.append({
            'user_id': f'user_{i+1:03d}',
            'name': generate_fake_name(),
            'persona': persona,
            'efficiency_rating': np.random.uniform(0.7, 1.3),
            **USER_PERSONAS[persona]
        })
    
    tasks_data = []
    dependencies_map = {}  # Track task dependencies
    project_tasks = {proj: [] for proj in PROJECTS}  # Group tasks by project
    
    # Generate tasks with realistic relationships
    for task_num in range(num_tasks):
        task_id = f'task_{task_num+1:05d}'
        
        # Assign to user and project
        user = np.random.choice(users)
        project = np.random.choice(PROJECTS)
        project_tasks[project].append(task_id)
        
        # Task type based on realistic distribution
        task_type = np.random.choice(list(TASK_TYPES.keys()), p=list(TASK_TYPES.values()))
        
        # Generate realistic start/due dates
        task_start = start_date + timedelta(days=np.random.randint(0, lookback_days + future_days))
        
        # Due date depends on task complexity and type
        complexity_days = {
            'coding': (3, 14), 'review': (1, 5), 'documentation': (2, 10),
            'meeting': (0, 3), 'testing': (2, 8), 'research': (5, 21),
            'planning': (1, 7), 'bug_fix': (1, 5), 'deployment': (1, 3), 'training': (3, 10)
        }
        
        min_days, max_days = complexity_days.get(task_type, (1, 7))
        due_date = task_start + timedelta(days=np.random.randint(min_days, max_days + 1))
        
        # Priority based on deadline proximity and task type
        days_to_deadline = (due_date - current_date).days
        if days_to_deadline < 1:
            priority_weights = [0.1, 0.2, 0.7]  # High priority for urgent
        elif days_to_deadline < 7:
            priority_weights = [0.2, 0.5, 0.3]  # Mixed for this week
        elif days_to_deadline < 30:
            priority_weights = [0.4, 0.4, 0.2]  # Lower for this month
        else:
            priority_weights = [0.6, 0.3, 0.1]  # Low for distant
        
        # High-impact task types get priority boost
        if task_type in ['bug_fix', 'deployment', 'meeting']:
            priority_weights = [p * 0.5 for p in priority_weights[:-1]] + [priority_weights[-1] * 1.5]
            # Normalize to sum to 1.0
            total = sum(priority_weights)
            priority_weights = [p / total for p in priority_weights]
            
        priority = np.random.choice([1, 2, 3], p=priority_weights)
        
        # Estimated time based on task type and complexity
        base_times = {
            'coding': (2, 8), 'review': (0.5, 3), 'documentation': (1, 6),
            'meeting': (0.5, 2), 'testing': (1, 4), 'research': (3, 12),
            'planning': (1, 4), 'bug_fix': (0.5, 4), 'deployment': (0.5, 2), 'training': (2, 8)
        }
        
        min_time, max_time = base_times.get(task_type, (1, 4))
        estimated_time = np.random.uniform(min_time, max_time)
        
        # Status based on timeline
        if task_start > current_date:
            status = 'todo'
            completion_date = None
        elif due_date < current_date - timedelta(days=5):
            # Older tasks are more likely completed
            status = np.random.choice(['completed', 'in_progress'], p=[0.8, 0.2])
            if status == 'completed':
                max_completion_days = max(1, (due_date - task_start).days + 3)
                completion_date = task_start + timedelta(days=np.random.randint(0, max_completion_days))
            else:
                completion_date = None
        else:
            # Recent tasks have mixed status
            status = np.random.choice(['completed', 'in_progress', 'todo'], p=[0.4, 0.3, 0.3])
            if status == 'completed':
                task_duration_days = (due_date - task_start).days
                if task_duration_days <= 0:
                    # Same day task - complete on start day
                    completion_date = task_start
                else:
                    # Complete within the task duration (0 to task_duration_days)
                    completion_days = np.random.randint(0, task_duration_days + 1)
                    completion_date = task_start + timedelta(days=completion_days)
        
        # Generate dependencies (10% of tasks have dependencies)
        dependencies = []
        if np.random.random() < 0.1 and len(tasks_data) > 5:
            # Pick 1-3 earlier tasks from same project as dependencies
            same_project_tasks = [t['task_id'] for t in tasks_data if t['project'] == project]
            if same_project_tasks:
                num_deps = min(3, max(1, len(same_project_tasks)))
                dependencies = np.random.choice(same_project_tasks, 
                                              size=min(num_deps, len(same_project_tasks)), 
                                              replace=False).tolist()
        
        # Generate realistic tags
        base_tags = [project, task_type]
        if priority == 3:
            base_tags.append('urgent')
        if dependencies:
            base_tags.append('dependent')
        if estimated_time > 6:
            base_tags.append('complex')
            
        # Add some random domain tags
        domain_tags = ['frontend', 'backend', 'database', 'ui', 'api', 'security', 'performance']
        base_tags.extend(np.random.choice(domain_tags, size=np.random.randint(0, 3), replace=False))
        
        task_data = {
            'task_id': task_id,
            'user_id': user['user_id'],
            'project': project,
            'type': task_type,
            'title': f"{task_type.replace('_', ' ').title()}: {generate_fake_phrase()}",
            'description': generate_fake_text(200),
            'priority': priority,
            'estimated_time': round(estimated_time, 2),
            'status': status,
            'start_date': task_start,
            'deadline': due_date,
            'completion_date': completion_date,
            'dependencies': json.dumps(dependencies),
            'tags': json.dumps(base_tags),
            'complexity_score': round(np.random.uniform(1, 10), 1)
        }
        
        tasks_data.append(task_data)
        if dependencies:
            dependencies_map[task_id] = dependencies
    
    # Generate realistic user behavior data
    behavior_data = []
    
    for task in tasks_data:
        if task['status'] in ['completed', 'in_progress']:
            user = next(u for u in users if u['user_id'] == task['user_id'])
            
            # Actual completion time varies by user efficiency and task complexity
            estimated = task['estimated_time']
            efficiency_factor = user['efficiency_rating']
            
            # Add realistic variation based on task type
            if task['type'] in ['coding', 'research']:  # More variable
                variation = np.random.normal(0, 0.4)
            else:  # More predictable
                variation = np.random.normal(0, 0.2)
            
            # Account for dependencies (dependent tasks take longer)
            dependency_penalty = 1.0
            if json.loads(task['dependencies']):
                dependency_penalty = 1.2
                
            actual_time = max(0.1, estimated * efficiency_factor * (1 + variation) * dependency_penalty)
            
            # Task switching patterns (some users switch more)
            switches_during_task = 0
            if not user['prefers_batch']:
                switches_during_task = np.random.poisson(1.5)
                actual_time *= (1 + switches_during_task * 0.15)  # Context switching overhead
            
            # Interruption patterns
            interruptions = np.random.poisson(0.8)  # Average interruptions
            actual_time *= (1 + interruptions * 0.1)
            
            # Work time patterns (some work outside normal hours)
            work_start, work_end = user['work_hours']
            worked_outside_hours = np.random.random() < 0.2
            if worked_outside_hours:
                actual_time *= 0.9  # More focused outside hours
            
            behavior_record = {
                'task_id': task['task_id'],
                'user_id': task['user_id'],
                'completion_time': round(actual_time, 2),
                'task_switches': switches_during_task,
                'interruptions': interruptions,
                'worked_outside_hours': worked_outside_hours,
                'actual_start_date': task['start_date'],
                'actual_completion_date': task['completion_date'],
                'efficiency_ratio': round(estimated / actual_time, 3) if actual_time > 0 else 0
            }
            
            behavior_data.append(behavior_record)
    
    return pd.DataFrame(tasks_data), pd.DataFrame(behavior_data), users

def generate_comprehensive_dataset(train_size=50000, test_size=2000, validation_size=1000):
    """Generate train/test/validation splits with comprehensive features"""
    
    print("Generating Enhanced Dataset for RL Training...")
    
    # Create directories
    for split in ['train', 'test', 'validation']:
        os.makedirs(f'data/enhanced/{split}', exist_ok=True)
    
    # Generate training data
    print(f"Generating training data ({train_size:,} tasks)...")
    train_tasks, train_behavior, users = generate_enhanced_dataset(
        num_tasks=train_size, 
        num_users=10,  # More users for training
        lookback_days=180,
        future_days=270
    )
    
    # Generate test data (different time period)
    print(f"Generating test data ({test_size:,} tasks)...")
    test_tasks, test_behavior, _ = generate_enhanced_dataset(
        num_tasks=test_size,
        num_users=3,  # Fewer users for test
        lookback_days=30,
        future_days=90
    )
    
    # Generate validation data
    print(f"Generating validation data ({validation_size:,} tasks)...")
    val_tasks, val_behavior, _ = generate_enhanced_dataset(
        num_tasks=validation_size,
        num_users=2,
        lookback_days=60,
        future_days=120
    )
    
    # Save datasets
    datasets = [
        ('train', train_tasks, train_behavior),
        ('test', test_tasks, test_behavior), 
        ('validation', val_tasks, val_behavior)
    ]
    
    for split_name, tasks_df, behavior_df in datasets:
        tasks_df.to_csv(f'data/enhanced/{split_name}/tasks.csv', index=False)
        behavior_df.to_csv(f'data/enhanced/{split_name}/user_behavior.csv', index=False)
        
        print(f"\n{split_name.upper()} DATA SUMMARY:")
        print(f"Tasks: {len(tasks_df):,}")
        print(f"Behavior records: {len(behavior_df):,}")
        print(f"Coverage: {len(behavior_df)/len(tasks_df)*100:.1f}%")
        
        print(f"Status distribution:")
        print(tasks_df['status'].value_counts(normalize=True).round(3))
        
        print(f"Task types: {tasks_df['type'].nunique()}")
        print(f"Users: {tasks_df['user_id'].nunique()}")
        print(f"Projects: {tasks_df['project'].nunique()}")
        
        # Temporal analysis
        current_date = datetime.now()
        tasks_df['deadline_dt'] = pd.to_datetime(tasks_df['deadline'])
        overdue = (tasks_df['deadline_dt'] < current_date).sum()
        future = (tasks_df['deadline_dt'] > current_date).sum()
        print(f"Temporal split: {overdue:,} overdue, {future:,} future")
    
    # Save user metadata
    users_df = pd.DataFrame(users)
    users_df.to_csv('data/enhanced/users.csv', index=False)
    
    print("\nEnhanced dataset generation complete!")
    print("Files created in data/enhanced/ directory")
    print("\nKey Improvements:")
    print("• Fresh temporal data (past & future)")
    print("• 10 realistic task types with proper distributions")
    print("• Multi-user scenarios with different personas") 
    print("• Task dependencies and project relationships")
    print("• Rich behavioral patterns (switching, interruptions)")
    print("• Comprehensive tags and metadata")
    print("• Realistic completion time variations")
    print("• Train/test/validation splits")

if __name__ == "__main__":
    generate_comprehensive_dataset()