import pandas as pd

def calculate_reward(task, user_behavior, current_task_index):
    """
    Calculate reward for prioritizing and completing a task.

    Args:
        task (pd.Series): The selected task data.
        user_behavior (pd.DataFrame): DataFrame with user behavior (completion time per task).
        current_task_index (int): Index of the current task in the sequence.

    Returns:
        float: Calculated reward.
    """
    task_id = task['task_id']
    priority = task['priority']
    status = task['status']
    deadline = pd.to_datetime(task['deadline'])
    estimated_time = task['estimated_time']
    
    # Check if task exists in user_behavior
    task_in_behavior = user_behavior['task_id'].isin([task_id]).any()
    
    if not task_in_behavior:
        # If task not in user_behavior, use a default approach
        return _default_reward(task, priority, status, deadline, estimated_time, current_task_index)
    
    # Get completion time from user_behavior
    completion_time = user_behavior.loc[
        user_behavior['task_id'] == task_id, 'completion_time'
    ].values[0]
    
    # Calculate base reward based on task status
    if status == 'completed':
        reward = _reward_for_completed_task(task, completion_time, estimated_time, priority, deadline)
    elif status == 'in_progress':
        reward = _reward_for_in_progress_task(task, user_behavior, completion_time, estimated_time, priority, deadline)
    else:  # 'todo'
        reward = _reward_for_todo_task(task, completion_time, estimated_time, priority, deadline)
    
    # Penalty for task switching to encourage focusing on similar priority tasks
    switching_penalty = -5 if current_task_index > 0 else 0
    
    return reward + switching_penalty

def _default_reward(task, priority, status, deadline, estimated_time, current_task_index):
    """Calculate a default reward when task is not in user_behavior."""
    start_date = pd.to_datetime(task['start_date'])
    days_remaining = (deadline - start_date).days
    
    # Base reward based on priority and status
    status_multiplier = {
        'completed': 2.0,
        'in_progress': 1.5,
        'todo': 1.0
    }
    
    # Deadline urgency (higher for closer deadlines)
    if days_remaining <= 7:
        urgency = 3.0
    elif days_remaining <= 14:
        urgency = 2.0
    elif days_remaining <= 30:
        urgency = 1.5
    else:
        urgency = 1.0
    
    # Calculate base reward
    base_reward = priority * 10 * status_multiplier[status] * urgency
    
    # Estimated time component (prefer shorter tasks)
    time_factor = max(0.5, 1.0 - (estimated_time / 10))
    
    # Switching penalty
    switching_penalty = -5 if current_task_index > 0 else 0
    
    return base_reward * time_factor + switching_penalty

def _reward_for_completed_task(task, completion_time, estimated_time, priority, deadline):
    """Calculate reward for a completed task."""
    # For completed tasks, use actual completion date vs deadline
    completion_date = pd.to_datetime(task['completion_date'])
    start_date = pd.to_datetime(task['start_date'])
    
    # Calculate actual days taken to complete
    days_taken = (completion_date - start_date).days
    
    # Calculate days to deadline (positive if completed before deadline)
    days_to_deadline = (deadline - completion_date).days
    
    # Base reward for completion
    base_reward = priority * 20  # Higher base reward for completed tasks
    
    # Deadline component
    if days_to_deadline >= 0:
        # Completed before deadline - bonus based on how early
        deadline_bonus = min(30, days_to_deadline * 2)
        deadline_component = 50 + deadline_bonus
    else:
        # Completed after deadline - penalty based on how late
        deadline_penalty = min(50, abs(days_to_deadline) * 3)
        deadline_component = -deadline_penalty
    
    # Time efficiency component
    if completion_time <= estimated_time:
        # Completed faster than estimated
        efficiency_bonus = 40 * (1 - (completion_time / estimated_time))
        time_component = 30 + efficiency_bonus
    else:
        # Took longer than estimated
        efficiency_penalty = 20 * ((completion_time / estimated_time) - 1)
        time_component = -efficiency_penalty
    
    return base_reward + deadline_component + time_component

def _reward_for_in_progress_task(task, user_behavior, completion_time, estimated_time, priority, deadline):
    """Calculate reward for an in-progress task."""
    # For in-progress tasks, use current progress and time spent
    start_date = pd.to_datetime(task['start_date'])
    
    # Check if actual_start_date exists in user_behavior
    if 'actual_start_date' in user_behavior.columns:
        actual_start_date = pd.to_datetime(user_behavior.loc[
            user_behavior['task_id'] == task['task_id'], 'actual_start_date'
        ].values[0])
    else:
        # Use start_date if actual_start_date not available
        actual_start_date = start_date
    
    # Use start_date as the current date for calculation
    current_date = start_date
    
    # Calculate days spent on task so far
    days_spent = max(1, (current_date - actual_start_date).days)
    
    # Calculate days remaining until deadline
    days_remaining = (deadline - current_date).days
    
    # Base reward scaled by priority
    base_reward = priority * 15  # Medium base reward for in-progress tasks
    
    # Deadline component
    if days_remaining > 0:
        # Still has time before deadline
        urgency_factor = 30 / max(1, days_remaining)  # Higher when closer to deadline
        deadline_component = 20 * urgency_factor
    else:
        # Past deadline but still working
        deadline_component = -30
    
    # Progress component - reward for making good progress
    # Assuming completion_time represents expected total time to complete
    expected_progress_ratio = min(1.0, days_spent / completion_time)
    
    if expected_progress_ratio < 0.5:
        # Early stages - small reward for being on track
        progress_component = 10
    elif expected_progress_ratio < 0.8:
        # Middle stages - medium reward
        progress_component = 25
    else:
        # Late stages - larger reward for almost complete tasks
        progress_component = 40
    
    # Efficiency component
    if days_spent < estimated_time:
        # On track to complete within estimated time
        efficiency_component = 15
    else:
        # Taking longer than estimated
        efficiency_component = -10
    
    return base_reward + deadline_component + progress_component + efficiency_component

def _reward_for_todo_task(task, completion_time, estimated_time, priority, deadline):
    """Calculate reward for a todo task."""
    # For todo tasks, focus on deadline and priority
    start_date = pd.to_datetime(task['start_date'])
    
    # Calculate days remaining until deadline
    days_remaining = (deadline - start_date).days
    
    # Base reward scaled by priority
    base_reward = priority * 10  # Lower base reward for todo tasks
    
    # Deadline component - higher reward for more urgent tasks
    if days_remaining <= 7:
        # Very urgent (within a week)
        deadline_component = 40
    elif days_remaining <= 14:
        # Urgent (within two weeks)
        deadline_component = 30
    elif days_remaining <= 30:
        # Moderately urgent (within a month)
        deadline_component = 20
    else:
        # Not urgent
        deadline_component = 10
    
    # Estimated time component - prefer shorter tasks when other factors are equal
    if estimated_time <= 1:
        # Very quick tasks
        time_component = 15
    elif estimated_time <= 3:
        # Medium-length tasks
        time_component = 10
    else:
        # Longer tasks
        time_component = 5
    
    return base_reward + deadline_component + time_component