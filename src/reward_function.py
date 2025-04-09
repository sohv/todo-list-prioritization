import pandas as pd

def calculate_reward(task, user_behavior, current_task_index):
    """
    calculate reward for prioritizing and completing a task.
    """
    task_id = task['task_id']
    priority = task['priority']
    status = task['status']
    deadline = pd.to_datetime(task['deadline'])
    estimated_time = task['estimated_time']
    
    task_in_behavior = user_behavior['task_id'].isin([task_id]).any()
    
    if not task_in_behavior:
        return _default_reward(task, priority, status, deadline, estimated_time, current_task_index)
    
    completion_time = user_behavior.loc[
        user_behavior['task_id'] == task_id, 'completion_time'
    ].values[0]
    
    if status == 'completed':
        reward = _reward_for_completed_task(task, completion_time, estimated_time, priority, deadline)
    elif status == 'in_progress':
        reward = _reward_for_in_progress_task(task, user_behavior, completion_time, estimated_time, priority, deadline)
    else:  # 'todo'
        reward = _reward_for_todo_task(task, completion_time, estimated_time, priority, deadline)
    
    # penalty for task switching to encourage focusing on similar priority tasks
    switching_penalty = -5 if current_task_index > 0 else 0
    
    return reward + switching_penalty

def _default_reward(task, priority, status, deadline, estimated_time, current_task_index):
    """calculate a default reward when task is not in user_behavior."""
    start_date = pd.to_datetime(task['start_date'])
    days_remaining = (deadline - start_date).days
    
    # base reward based on priority and status
    status_multiplier = {
        'completed': 2.0,
        'in_progress': 1.5,
        'todo': 1.0
    }
    
    if days_remaining <= 7:
        urgency = 3.0
    elif days_remaining <= 14:
        urgency = 2.0
    elif days_remaining <= 30:
        urgency = 1.5
    else:
        urgency = 1.0
    
    # calculate base reward
    base_reward = priority * 10 * status_multiplier[status] * urgency
    # estimate time component
    time_factor = max(0.5, 1.0 - (estimated_time / 10))
    switching_penalty = -5 if current_task_index > 0 else 0
    return base_reward * time_factor + switching_penalty

def _reward_for_completed_task(task, completion_time, estimated_time, priority, deadline):
    """calculate reward for a completed task."""
    completion_date = pd.to_datetime(task['completion_date'])
    start_date = pd.to_datetime(task['start_date'])
    days_taken = (completion_date - start_date).days
    days_to_deadline = (deadline - completion_date).days # calculate days to deadline
    base_reward = priority * 20  # higher base reward for completed tasks
    
    if days_to_deadline >= 0:
        deadline_bonus = min(30, days_to_deadline * 2)
        deadline_component = 50 + deadline_bonus
    else:
        deadline_penalty = min(50, abs(days_to_deadline) * 3)
        deadline_component = -deadline_penalty
    
    if completion_time <= estimated_time:
        efficiency_bonus = 40 * (1 - (completion_time / estimated_time))
        time_component = 30 + efficiency_bonus
    else:
        efficiency_penalty = 20 * ((completion_time / estimated_time) - 1)
        time_component = -efficiency_penalty
    
    return base_reward + deadline_component + time_component

def _reward_for_in_progress_task(task, user_behavior, completion_time, estimated_time, priority, deadline):
    """calculate reward for an in-progress task."""
    start_date = pd.to_datetime(task['start_date'])
    
    if 'actual_start_date' in user_behavior.columns:
        actual_start_date = pd.to_datetime(user_behavior.loc[
            user_behavior['task_id'] == task['task_id'], 'actual_start_date'
        ].values[0])
    else:
        actual_start_date = start_date
    
    current_date = start_date
    days_spent = max(1, (current_date - actual_start_date).days)
    days_remaining = (deadline - current_date).days
    base_reward = priority * 15  # medium base reward for in-progress tasks
    
    if days_remaining > 0:
        urgency_factor = 30 / max(1, days_remaining)  # higher reward when closer to deadline
        deadline_component = 20 * urgency_factor
    else:
        deadline_component = -30
    
    # reward for making good progress
    expected_progress_ratio = min(1.0, days_spent / completion_time)
    
    if expected_progress_ratio < 0.5:
        progress_component = 10
    elif expected_progress_ratio < 0.8:
        progress_component = 25
    else:
        progress_component = 40
    
    if days_spent < estimated_time:
        efficiency_component = 15
    else:
        efficiency_component = -10
    
    return base_reward + deadline_component + progress_component + efficiency_component

def _reward_for_todo_task(task, completion_time, estimated_time, priority, deadline):
    """calculate reward for a todo task."""
    start_date = pd.to_datetime(task['start_date'])
    days_remaining = (deadline - start_date).days
    base_reward = priority * 10  # lower base reward for todo tasks
    if days_remaining <= 7:
        deadline_component = 40
    elif days_remaining <= 14:
        deadline_component = 30
    elif days_remaining <= 30:
        deadline_component = 20
    else:
        deadline_component = 10
    
    if estimated_time <= 1:
        time_component = 15
    elif estimated_time <= 3:
        time_component = 10
    else:
        time_component = 5
    
    return base_reward + deadline_component + time_component