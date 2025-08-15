import pandas as pd
import numpy as np
from datetime import datetime

def calculate_reward(task, user_behavior, current_step, available_tasks=None, task_history=None):
    """
    Calculate simplified reward for prioritizing and completing a task.
    Returns reward in range [-1, 1] focusing on core factors only.
    """
    # Core components only - keep it simple
    priority_score = _calculate_priority_score(task)
    urgency_score = _calculate_urgency_score(task)
    efficiency_score = _calculate_efficiency_score(task, user_behavior)
    
    # Simple weighted combination - no over-engineering
    reward = (
        0.4 * priority_score +     # Task importance
        0.4 * urgency_score +      # Deadline pressure  
        0.2 * efficiency_score     # Time estimation
    )
    
    return np.clip(reward, -1.0, 1.0)

def _calculate_priority_score(task):
    """Simple priority scoring: 1-3 scale mapped to [-1, 1]"""
    priority = task.get('priority', 2)
    
    # Direct linear mapping - no over-engineering
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
        
        # Simple urgency bands - no over-engineering
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

# Removed all over-engineered components:
# - _calculate_context_score (complex dependency and batching logic)
# - _calculate_status_score (status-aware bonuses) 
# - _calculate_exploration_score (exploration bonuses)
# - _calculate_penalty_score (multiple penalty patterns)
# - update_user_preferences (preference learning)
# - get_preference_learner (global preference instance)
#
# The reward function now focuses only on the core factors:
# priority, urgency, and efficiency - keeping it simple and effective.