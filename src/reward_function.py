import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
import hashlib

class UserPreferenceLearner:
    def __init__(self):
        self.preference_history = defaultdict(lambda: deque(maxlen=50))
        self.task_type_history = defaultdict(lambda: deque(maxlen=100))
        self.pattern_violations = defaultdict(int)
    
    def update_preferences(self, task, reward_outcome):
        user_id = task.get('user_id', 'default')
        task_type = task.get('type', 'general')
        priority = task['priority']
        
        self.preference_history[f"{user_id}_priority_{priority}"].append(reward_outcome)
        self.task_type_history[f"{user_id}_type_{task_type}"].append(reward_outcome)
    
    def get_preference_score(self, task):
        user_id = task.get('user_id', 'default')
        task_type = task.get('type', 'general')
        priority = task['priority']
        
        priority_history = self.preference_history[f"{user_id}_priority_{priority}"]
        type_history = self.task_type_history[f"{user_id}_type_{task_type}"]
        
        priority_pref = np.mean(priority_history) if priority_history else 0.0
        type_pref = np.mean(type_history) if type_history else 0.0
        
        return np.clip((priority_pref + type_pref) / 2.0, -0.5, 0.5)

# Global preference learner instance
_preference_learner = UserPreferenceLearner()

def calculate_reward(task, user_behavior, current_step, available_tasks=None, task_history=None):
    """
    Calculate comprehensive reward for prioritizing and completing a task.
    Returns reward in range [-1, 1] with full utilization of the space.
    """
    # Core components with expanded ranges
    priority_score = _calculate_priority_score(task)
    urgency_score = _calculate_urgency_score(task)
    efficiency_score = _calculate_efficiency_score(task, user_behavior)
    context_score = _calculate_context_score(task, available_tasks, current_step, task_history)
    
    # New advanced components
    status_score = _calculate_status_score(task, available_tasks)
    preference_score = _preference_learner.get_preference_score(task)
    exploration_score = _calculate_exploration_score(task, task_history)
    penalty_score = _calculate_penalty_score(task, user_behavior, task_history)
    
    # Optimized weighted combination to maximize range utilization
    base_reward = (
        0.35 * priority_score +      # Task importance (increased weight)
        0.35 * urgency_score +       # Deadline pressure (increased weight)
        0.12 * efficiency_score +    # Time estimation accuracy
        0.08 * context_score +       # Enhanced context and sequencing
        0.10 * status_score          # Status-aware bonuses (direct addition)
    )
    
    # Add enhancement bonuses/penalties
    enhancements = (
        preference_score +           # User preference learning
        exploration_score +          # Exploration bonuses
        penalty_score               # Negative patterns
    )
    
    reward = base_reward + enhancements
    
    # Apply non-linear tanh scaling to better utilize the full [-1, 1] range
    # This stretches the distribution toward the extremes for stronger learning signals
    reward = np.tanh(1.5 * reward)
    
    return np.clip(reward, -1.0, 1.0)

def _calculate_priority_score(task):
    """Priority contribution with full [-1, 1] range utilization"""
    priority = task['priority']
    
    # Enhanced priority scoring with full range
    if priority == 3:
        return 1.0      # High priority: maximum positive
    elif priority == 2:
        return 0.0      # Medium priority: neutral
    elif priority == 1:
        return -1.0     # Low priority: maximum negative
    else:
        # Handle edge cases
        normalized = np.clip((priority - 2) / 1.0, -1.0, 1.0)
        return normalized

def _calculate_urgency_score(task):
    """Enhanced urgency scoring with full [-1, 1] range utilization"""
    current_date = pd.to_datetime(datetime.now())
    deadline = pd.to_datetime(task['deadline'])
    days_remaining = (deadline - current_date).days
    
    # Expanded urgency ranges for better learning signals
    if days_remaining <= -7:
        return 1.0      # Severely overdue
    elif days_remaining <= -1:
        return 0.9      # Overdue by days
    elif days_remaining <= 0:
        return 0.8      # Due today or just passed
    elif days_remaining <= 1:
        return 0.6      # Due tomorrow
    elif days_remaining <= 3:
        return 0.3      # Due this week
    elif days_remaining <= 7:
        return 0.0      # Due next week (neutral)
    elif days_remaining <= 14:
        return -0.3     # Due in 2 weeks
    elif days_remaining <= 30:
        return -0.6     # Due this month
    else:
        return -1.0     # Not urgent at all

def _calculate_efficiency_score(task, user_behavior):
    """Enhanced efficiency scoring with full [-1, 1] range utilization"""
    task_id = task['task_id']
    estimated_time = task.get('estimated_time', 1.0)
    
    # Check if we have historical data for this task
    if user_behavior.empty or not user_behavior['task_id'].isin([task_id]).any():
        # No historical data - score based on estimated time and task characteristics
        time_factor = np.clip(1.0 - (estimated_time / 10.0), -1.0, 1.0)
        return time_factor * 0.3  # Conservative scoring without data
    
    completion_time = user_behavior.loc[
        user_behavior['task_id'] == task_id, 'completion_time'
    ].values[0]
    
    if pd.isna(completion_time) or completion_time <= 0:
        return 0.0  # No valid completion data
    
    # Enhanced efficiency scoring with full range utilization
    efficiency_ratio = estimated_time / completion_time
    
    if efficiency_ratio >= 2.0:        # Much faster than expected
        return 1.0
    elif efficiency_ratio >= 1.5:      # Significantly faster
        return 0.7
    elif efficiency_ratio >= 1.2:      # Faster than expected
        return 0.4
    elif efficiency_ratio >= 0.9:      # Close to estimate (good)
        return 0.2
    elif efficiency_ratio >= 0.7:      # Slightly slower
        return -0.2
    elif efficiency_ratio >= 0.5:      # Much slower
        return -0.6
    else:                               # Severely slower
        return -1.0

def _calculate_context_score(task, available_tasks, current_step, task_history=None):
    """Enhanced context-aware scoring with dependency and batching logic"""
    if available_tasks is None or len(available_tasks) <= 1:
        return 0.0
    
    score = 0.0
    current_priority = task['priority']
    current_type = task.get('type', 'general')
    current_deadline = pd.to_datetime(task['deadline'])
    
    # Dependency scoring
    dependencies = task.get('dependencies', [])
    if dependencies:
        completed_deps = 0
        for dep_id in dependencies:
            if task_history and any(h.get('task_id') == dep_id for h in task_history):
                completed_deps += 1
        
        if completed_deps == len(dependencies):
            score += 0.5  # All dependencies completed
        elif completed_deps > 0:
            score += 0.2 * (completed_deps / len(dependencies))
        else:
            score -= 0.8  # Dependencies not met - strong penalty
    
    # Enhanced batching logic
    if isinstance(available_tasks, pd.DataFrame) and len(available_tasks) > 1:
        # Priority batching
        similar_priority_count = len(available_tasks[available_tasks['priority'] == current_priority])
        if similar_priority_count > 1:
            score += 0.3 * min(similar_priority_count / len(available_tasks), 0.5)
        
        # Task type batching
        if 'type' in available_tasks.columns:
            similar_type_count = len(available_tasks[available_tasks['type'] == current_type])
            if similar_type_count > 1:
                score += 0.2 * min(similar_type_count / len(available_tasks), 0.4)
        
        # Deadline clustering
        available_deadlines = pd.to_datetime(available_tasks['deadline'])
        deadline_diff = abs((available_deadlines - current_deadline).dt.days)
        nearby_deadlines = sum(deadline_diff <= 2)
        if nearby_deadlines > 1:
            score += 0.25 * min(nearby_deadlines / len(available_tasks), 0.4)
    
    # Exploration encouragement for early steps
    if current_step < 3 and len(available_tasks) > 5:
        exploration_bonus = 0.1 * (3 - current_step) / 3
        score += exploration_bonus
    
    return np.clip(score, -1.0, 1.0)

def _calculate_status_score(task, available_tasks):
    """Status-aware bonuses for prioritizing in-progress tasks"""
    current_status = task.get('status', 'pending').lower()
    
    # Strong incentive to complete in-progress tasks
    if current_status == 'in_progress':
        return 1.0
    elif current_status == 'blocked':
        return -0.8  # Avoid blocked tasks
    elif current_status == 'pending':
        # Check if there are in-progress tasks that should be prioritized
        if available_tasks is not None and isinstance(available_tasks, pd.DataFrame):
            if 'status' in available_tasks.columns:
                in_progress_count = len(available_tasks[available_tasks['status'] == 'in_progress'])
                if in_progress_count > 0:
                    return -0.4  # Penalize starting new tasks when others are in progress
        return 0.0  # Neutral for pending tasks
    elif current_status == 'completed':
        return -1.0  # Should not select completed tasks
    else:
        return 0.0  # Unknown status

def _calculate_exploration_score(task, task_history):
    """Exploration bonuses for trying different task types"""
    if not task_history:
        return 0.2  # Encourage exploration when no history
    
    current_type = task.get('type', 'general')
    current_priority = task['priority']
    
    # Analyze recent task history (last 10 tasks)
    recent_history = task_history[-10:] if len(task_history) > 10 else task_history
    
    # Task type diversity
    recent_types = [t.get('type', 'general') for t in recent_history]
    type_frequency = recent_types.count(current_type) / len(recent_types) if recent_types else 0
    
    # Priority diversity
    recent_priorities = [t.get('priority', 2) for t in recent_history]
    priority_frequency = recent_priorities.count(current_priority) / len(recent_priorities) if recent_priorities else 0
    
    # Exploration bonus: higher bonus for less frequently chosen types/priorities
    type_exploration = 0.5 * (1.0 - type_frequency)
    priority_exploration = 0.3 * (1.0 - priority_frequency)
    
    exploration_score = type_exploration + priority_exploration
    
    # Add bonus for trying new task types
    unique_types = set(recent_types)
    if current_type not in unique_types:
        exploration_score += 0.4  # Bonus for completely new type
    
    return np.clip(exploration_score, -0.2, 1.0)

def _calculate_penalty_score(task, user_behavior, task_history):
    """Negative rewards for poor prioritization patterns"""
    penalty = 0.0
    
    # Deadline miss penalty
    current_date = pd.to_datetime(datetime.now())
    deadline = pd.to_datetime(task['deadline'])
    if deadline < current_date:
        days_overdue = (current_date - deadline).days
        penalty -= min(0.5 + 0.1 * days_overdue, 1.0)  # Escalating penalty
    
    # Task switching penalty
    if task_history and len(task_history) > 1:
        last_task = task_history[-1]
        if (last_task.get('status') == 'in_progress' and 
            last_task.get('task_id') != task.get('task_id')):
            penalty -= 0.3  # Penalty for abandoning in-progress tasks
    
    # Priority inconsistency penalty
    if task_history and len(task_history) >= 3:
        recent_priorities = [t.get('priority', 2) for t in task_history[-3:]]
        priority_variance = np.var(recent_priorities)
        if priority_variance > 1.0:  # High variance in recent priorities
            penalty -= 0.2
    
    # Inefficiency pattern penalty
    if not user_behavior.empty:
        task_id = task.get('task_id')
        user_tasks = user_behavior[user_behavior['task_id'] == task_id]
        if not user_tasks.empty:
            avg_completion_time = user_tasks['completion_time'].mean()
            estimated_time = task.get('estimated_time', 1.0)
            if avg_completion_time > 2 * estimated_time:
                penalty -= 0.25  # Penalty for consistently slow tasks
    
    # Procrastination penalty (selecting low-urgency tasks when high-urgency exist)
    if task.get('priority', 2) == 1:  # Low priority task
        urgency_score = _calculate_urgency_score(task)
        if urgency_score < 0:  # Not urgent
            penalty -= 0.15  # Small penalty for procrastination
    
    return np.clip(penalty, -1.0, 0.0)  # Penalties are always negative or zero

# Export functions for external preference learning updates
def update_user_preferences(task, reward_outcome):
    """Update user preference learning with task outcome"""
    _preference_learner.update_preferences(task, reward_outcome)

def get_preference_learner():
    """Get the global preference learner instance"""
    return _preference_learner