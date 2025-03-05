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
    days_remaining = task['days_remaining']
    priority = task['priority']

    # Closer deadlines are more urgent (capped to 30 days)
    deadline_urgency = max(1, 30 - days_remaining)

    # Lookup the completion time from user behavior
    completion_time = user_behavior.loc[
        user_behavior['task_id'] == task['task_id'], 'completion_time'
    ].values[0]

    # Base reward for prioritization (higher priority + closer deadline = more reward)
    prioritization_reward = (priority * 5) + deadline_urgency

    # Bonus or penalty based on whether the task is completed on time
    if completion_time <= days_remaining:
        completion_bonus = priority * 10  # Big reward for on-time completion
    else:
        completion_bonus = -priority * 5  # Big penalty for lateness

    switching_penalty = -1 if current_task_index > 0 else 0

    reward = prioritization_reward + completion_bonus + switching_penalty

    return reward
