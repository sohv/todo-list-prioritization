#!/usr/bin/env python3
"""
Enhanced Reward Function - Production Ready Implementation

This example demonstrates the comprehensive reward function with all 6 improvements:
1. Full [-1, 1] range utilization
2. Status-aware bonuses
3. User preference learning  
4. Enhanced context with dependencies & batching
5. Exploration bonuses for task diversity
6. Negative penalties for poor patterns

Usage Example:
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from reward_function import calculate_reward, update_user_preferences

def example_usage():
    """Demonstrate enhanced reward function capabilities"""
    
    print("Enhanced Reward Function - Production Example")
    print("=" * 55)
    
    # Sample user behavior data
    user_behavior = pd.DataFrame([
        {'task_id': 'task_001', 'completion_time': 2.5, 'user_id': 'user_alice'},
        {'task_id': 'task_002', 'completion_time': 1.0, 'user_id': 'user_alice'},
        {'task_id': 'task_003', 'completion_time': 4.5, 'user_id': 'user_alice'},
    ])
    
    # Available tasks with rich metadata
    available_tasks = pd.DataFrame([
        {
            'task_id': 'task_100', 'priority': 3, 'type': 'coding', 
            'deadline': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'status': 'in_progress', 'estimated_time': 3.0
        },
        {
            'task_id': 'task_101', 'priority': 3, 'type': 'review', 
            'deadline': (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'),
            'status': 'pending', 'estimated_time': 1.5
        },
        {
            'task_id': 'task_102', 'priority': 2, 'type': 'documentation', 
            'deadline': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            'status': 'pending', 'estimated_time': 2.0
        },
        {
            'task_id': 'task_103', 'priority': 1, 'type': 'research', 
            'deadline': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'status': 'blocked', 'estimated_time': 5.0
        }
    ])
    
    # Task history showing recent activity
    task_history = [
        {'task_id': 'task_090', 'type': 'coding', 'priority': 3, 'status': 'completed'},
        {'task_id': 'task_091', 'type': 'coding', 'priority': 3, 'status': 'completed'},
        {'task_id': 'task_092', 'type': 'review', 'priority': 2, 'status': 'completed'},
    ]
    
    print("Available Tasks:")
    print("-" * 25)
    
    # Calculate rewards for each task
    for idx, task_row in available_tasks.iterrows():
        task = task_row.to_dict()
        
        reward = calculate_reward(
            task=task,
            user_behavior=user_behavior,
            current_step=len(task_history),
            available_tasks=available_tasks,
            task_history=task_history
        )
        
        print(f"Task {task['task_id']}:")
        print(f"  Priority: {task['priority']}, Type: {task['type']}")
        print(f"  Status: {task['status']}, Due: {task['deadline']}")
        print(f"  Reward Score: {reward:.3f}")
        
        # Show reward breakdown
        if reward > 0.5:
            print("  HIGH PRIORITY - Strongly recommended")
        elif reward > 0.3:
            print("  GOOD CHOICE - Recommended")
        elif reward > 0.0:
            print("  NEUTRAL - Consider alternatives")
        else:
            print("  LOW PRIORITY - Avoid unless necessary")
        print()
    
    # Demonstrate preference learning
    print("🧠 User Preference Learning:")
    print("-" * 30)
    
    # Simulate learning from successful task completions
    coding_task = {'task_id': 'demo', 'priority': 3, 'type': 'coding', 'user_id': 'user_alice'}
    documentation_task = {'task_id': 'demo', 'priority': 2, 'type': 'documentation', 'user_id': 'user_alice'}
    
    print("Simulating preference learning...")
    # User has good outcomes with coding tasks
    for _ in range(15):
        update_user_preferences(coding_task, 0.8)  # Positive outcome
    
    # User has mixed outcomes with documentation
    for _ in range(10):
        update_user_preferences(documentation_task, 0.3)  # Neutral outcome
    
    print("Preferences learned from task history")
    print("\nKey Features Demonstrated:")
    print("   Full range utilization: [-1.0, +1.0]")
    print("   Status-aware prioritization (in-progress > pending > blocked)")
    print("   🧠 Personalized preference learning")
    print("   🔗 Dependency and batching awareness")
    print("   Task diversity exploration bonuses")
    print("   Poor prioritization pattern penalties")
    
    # Show optimal task selection
    rewards = []
    task_ids = []
    
    for _, task_row in available_tasks.iterrows():
        task = task_row.to_dict()
        reward = calculate_reward(task, user_behavior, 0, available_tasks, task_history)
        rewards.append(reward)
        task_ids.append(task['task_id'])
    
    optimal_idx = np.argmax(rewards)
    optimal_task = available_tasks.iloc[optimal_idx]
    
    print(f"\nRECOMMENDED TASK: {optimal_task['task_id']}")
    print(f"   Reward Score: {rewards[optimal_idx]:.3f}")
    print(f"   Reason: {optimal_task['status']} {optimal_task['type']} task")
    print(f"   Priority: {optimal_task['priority']}, Due: {optimal_task['deadline']}")

if __name__ == "__main__":
    example_usage()