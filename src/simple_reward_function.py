"""
Simple Reward Function - Basic Implementation

This is a simplified version of the reward function suitable for:
- Initial prototyping and baseline comparisons
- Applications where complexity isn't warranted
- Learning systems that need gradual complexity introduction
- Quick implementation and testing

Recommendation: Start with this version and gradually add components
from the advanced reward function based on learning performance needs.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_simple_reward(task, user_behavior=None, current_step=0):
    """
    Calculate basic reward for task prioritization.
    Simple 3-component model focusing on core factors.
    
    Returns reward in range [-1, 1]
    """
    # Core components only
    priority_score = _calculate_priority_score(task)
    urgency_score = _calculate_urgency_score(task)
    efficiency_score = _calculate_efficiency_score(task, user_behavior)
    
    # Simple weighted combination
    reward = (
        0.4 * priority_score +     # Task importance
        0.4 * urgency_score +      # Deadline pressure  
        0.2 * efficiency_score     # Time estimation
    )
    
    return np.clip(reward, -1.0, 1.0)

def _calculate_priority_score(task):
    """Simple priority scoring: 1-3 scale mapped to [-1, 1]"""
    priority = task.get('priority', 2)
    
    # Direct linear mapping
    if priority == 3:
        return 1.0      # High priority
    elif priority == 2:
        return 0.0      # Medium priority
    else:  # priority == 1
        return -1.0     # Low priority

def _calculate_urgency_score(task):
    """Basic urgency scoring based on deadline proximity"""
    try:
        current_date = pd.to_datetime(datetime.now())
        deadline = pd.to_datetime(task['deadline'])
        days_remaining = (deadline - current_date).days
        
        # Simple urgency bands
        if days_remaining <= 0:
            return 1.0      # Overdue
        elif days_remaining <= 1:
            return 0.7      # Due soon
        elif days_remaining <= 7:
            return 0.3      # Due this week
        elif days_remaining <= 30:
            return 0.0      # Due this month
        else:
            return -0.5     # Not urgent
            
    except (KeyError, ValueError):
        return 0.0  # Default if no valid deadline

def _calculate_efficiency_score(task, user_behavior):
    """Basic efficiency scoring based on estimated time"""
    if user_behavior is None or user_behavior.empty:
        # No history - prefer shorter tasks
        estimated_time = task.get('estimated_time', 2.0)
        return np.clip(0.5 - (estimated_time / 10.0), -0.5, 0.5)
    
    task_id = task.get('task_id')
    if task_id and task_id in user_behavior['task_id'].values:
        # Use historical completion time
        completion_time = user_behavior[
            user_behavior['task_id'] == task_id
        ]['completion_time'].iloc[0]
        
        estimated_time = task.get('estimated_time', 2.0)
        
        if completion_time > 0:
            efficiency_ratio = estimated_time / completion_time
            
            if efficiency_ratio > 1.2:     # Much faster
                return 0.5
            elif efficiency_ratio > 0.8:   # Close to estimate
                return 0.2
            else:                          # Slower
                return -0.3
    
    return 0.0  # Default neutral score

def compare_tasks(tasks, user_behavior=None):
    """
    Compare multiple tasks and return them sorted by reward score.
    
    Args:
        tasks: List of task dictionaries
        user_behavior: Optional pandas DataFrame with historical data
        
    Returns:
        List of (task, reward_score) tuples sorted by score (highest first)
    """
    scored_tasks = []
    
    for i, task in enumerate(tasks):
        reward = calculate_simple_reward(task, user_behavior, i)
        scored_tasks.append((task, reward))
    
    # Sort by reward score (highest first)
    scored_tasks.sort(key=lambda x: x[1], reverse=True)
    
    return scored_tasks

def get_best_task(tasks, user_behavior=None):
    """
    Get the single best task from a list based on reward score.
    
    Returns:
        tuple: (best_task, reward_score)
    """
    if not tasks:
        return None, 0.0
    
    scored_tasks = compare_tasks(tasks, user_behavior)
    return scored_tasks[0]

# Utility functions for gradual complexity introduction
def add_status_bonus(base_reward, task):
    """
    Optional: Add simple status-aware bonus to base reward.
    Use this to gradually introduce status awareness.
    """
    status = task.get('status', 'pending').lower()
    
    if status == 'in_progress':
        return base_reward + 0.2  # Small bonus for continuing work
    elif status == 'blocked':
        return base_reward - 0.3  # Penalty for blocked tasks
    
    return base_reward

def add_overdue_penalty(base_reward, task):
    """
    Optional: Add penalty for significantly overdue tasks.
    Use this to introduce pattern-based penalties gradually.
    """
    try:
        current_date = pd.to_datetime(datetime.now())
        deadline = pd.to_datetime(task['deadline'])
        days_overdue = (current_date - deadline).days
        
        if days_overdue > 0:  # Any overdue amount
            penalty = min(0.5, days_overdue * 0.1)  # Escalating penalty
            return base_reward - penalty
            
    except (KeyError, ValueError):
        pass
    
    return base_reward

# Example usage and migration path
def calculate_progressive_reward(task, user_behavior=None, current_step=0, 
                               include_status=False, include_penalties=False):
    """
    Progressive reward function that allows gradual complexity introduction.
    
    Args:
        task: Task dictionary
        user_behavior: Optional historical data
        current_step: Current step in task sequence
        include_status: Whether to include status-aware bonuses
        include_penalties: Whether to include penalty patterns
        
    Returns:
        Reward score in [-1, 1] range
    """
    # Start with base reward
    reward = calculate_simple_reward(task, user_behavior, current_step)
    
    # Gradually add complexity
    if include_status:
        reward = add_status_bonus(reward, task)
    
    if include_penalties:
        reward = add_overdue_penalty(reward, task)
    
    return np.clip(reward, -1.0, 1.0)