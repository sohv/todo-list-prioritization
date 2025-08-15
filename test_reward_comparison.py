#!/usr/bin/env python3
"""
Comprehensive Reward Function Comparison and Analysis Tool

This script provides detailed analysis of reward functions to help debug
negative reward issues and validate reward function improvements.
"""

import sys
import os
sys.path.append('/root/todo/src')
sys.path.append('/root/todo')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import reward functions
from src.reward_function import calculate_reward
from src.simple_reward_function import calculate_simple_reward

def generate_realistic_tasks(n_tasks=100):
    """Generate realistic task scenarios for testing"""
    tasks = []
    base_date = datetime.now()
    
    # Task type distributions (realistic)
    priorities = [1, 2, 3]  # Low, Medium, High
    priority_weights = [0.3, 0.5, 0.2]  # Most tasks are medium priority
    
    task_types = ['work', 'personal', 'urgent', 'project', 'maintenance']
    statuses = ['pending', 'in_progress', 'blocked']
    status_weights = [0.7, 0.2, 0.1]
    
    for i in range(n_tasks):
        # Realistic distributions
        priority = np.random.choice(priorities, p=priority_weights)
        
        # Deadline varies by priority
        if priority == 3:  # High priority - often urgent
            days_offset = np.random.randint(-3, 14)
        elif priority == 2:  # Medium priority - normal timeline
            days_offset = np.random.randint(-1, 30)
        else:  # Low priority - can be distant
            days_offset = np.random.randint(5, 60)
        
        deadline = base_date + timedelta(days=days_offset)
        
        # Estimated time varies by priority and type
        if priority == 3:
            estimated_time = np.random.uniform(0.5, 4.0)  # High priority often quick
        else:
            estimated_time = np.random.uniform(1.0, 8.0)
        
        task = {
            'task_id': f'task_{i:03d}',
            'priority': priority,
            'deadline': deadline,
            'estimated_time': estimated_time,
            'type': np.random.choice(task_types),
            'status': np.random.choice(statuses, p=status_weights),
            'start_date': base_date
        }
        tasks.append(task)
    
    return tasks

def generate_user_behavior(tasks, completion_rate=0.6):
    """Generate realistic user behavior data"""
    behaviors = []
    
    for task in tasks:
        if np.random.random() < completion_rate:
            # Realistic completion time based on estimate with variance
            base_time = task['estimated_time']
            
            # Add realistic variance (some tasks take longer/shorter)
            variance_factor = np.random.uniform(0.5, 2.5)
            completion_time = base_time * variance_factor
            
            behaviors.append({
                'task_id': task['task_id'],
                'completion_time': completion_time,
                'completed_date': task['deadline'] - timedelta(days=np.random.randint(0, 5))
            })
    
    return pd.DataFrame(behaviors)

def compare_reward_functions():
    """Compare simplified vs simple reward functions comprehensively"""
    print("🔍 GENERATING TEST DATA...")
    tasks = generate_realistic_tasks(200)
    user_behavior = generate_user_behavior(tasks)
    available_tasks = pd.DataFrame(tasks)
    
    print(f"📊 Generated {len(tasks)} tasks with {len(user_behavior)} completion records")
    print()
    
    # Calculate rewards with both functions
    print("⚡ CALCULATING REWARDS...")
    simplified_rewards = []  # Our new simplified version
    simple_rewards = []      # The simple_reward_function.py version
    
    for i, task in enumerate(tasks[:100]):  # Test first 100 tasks
        # Simplified reward (current reward_function.py)
        simplified_reward = calculate_reward(
            task, user_behavior, i, available_tasks
        )
        simplified_rewards.append(simplified_reward)
        
        # Simple reward (simple_reward_function.py)
        simple_reward = calculate_simple_reward(
            task, user_behavior, i
        )
        simple_rewards.append(simple_reward)
    
    # Statistical analysis
    print("📈 STATISTICAL ANALYSIS")
    print("=" * 50)
    
    print(f"Simplified Reward Function (reward_function.py):")
    print(f"  Mean:     {np.mean(simplified_rewards):7.3f}")
    print(f"  Median:   {np.median(simplified_rewards):7.3f}")
    print(f"  Std Dev:  {np.std(simplified_rewards):7.3f}")
    print(f"  Min:      {np.min(simplified_rewards):7.3f}")
    print(f"  Max:      {np.max(simplified_rewards):7.3f}")
    print(f"  Negative: {(np.array(simplified_rewards) < 0).sum()}/{len(simplified_rewards)} ({(np.array(simplified_rewards) < 0).mean()*100:.1f}%)")
    
    print()
    print(f"Simple Reward Function (simple_reward_function.py):")
    print(f"  Mean:     {np.mean(simple_rewards):7.3f}")
    print(f"  Median:   {np.median(simple_rewards):7.3f}")
    print(f"  Std Dev:  {np.std(simple_rewards):7.3f}")
    print(f"  Min:      {np.min(simple_rewards):7.3f}")
    print(f"  Max:      {np.max(simple_rewards):7.3f}")
    print(f"  Negative: {(np.array(simple_rewards) < 0).sum()}/{len(simple_rewards)} ({(np.array(simple_rewards) < 0).mean()*100:.1f}%)")
    
    # Category analysis
    print()
    print("📊 REWARD BY PRIORITY ANALYSIS")
    print("=" * 50)
    
    for priority in [1, 2, 3]:
        priority_tasks = [i for i, task in enumerate(tasks[:100]) if task['priority'] == priority]
        if priority_tasks:
            simp_rewards_p = [simplified_rewards[i] for i in priority_tasks]
            simple_rewards_p = [simple_rewards[i] for i in priority_tasks]
            
            print(f"Priority {priority} tasks ({len(priority_tasks)} tasks):")
            print(f"  Simplified: {np.mean(simp_rewards_p):6.3f} ± {np.std(simp_rewards_p):5.3f}")
            print(f"  Simple:     {np.mean(simple_rewards_p):6.3f} ± {np.std(simple_rewards_p):5.3f}")
    
    # Urgency analysis
    print()
    print("📊 REWARD BY URGENCY ANALYSIS")
    print("=" * 50)
    
    current_date = datetime.now()
    for urgency_desc, day_range in [
        ("Overdue", (-999, 0)),
        ("Due soon (1-7 days)", (1, 7)),
        ("Due later (8-30 days)", (8, 30)),
        ("Not urgent (30+ days)", (31, 999))
    ]:
        urgency_tasks = []
        for i, task in enumerate(tasks[:100]):
            days_remaining = (task['deadline'] - current_date).days
            if day_range[0] <= days_remaining <= day_range[1]:
                urgency_tasks.append(i)
        
        if urgency_tasks:
            simp_rewards_u = [simplified_rewards[i] for i in urgency_tasks]
            simple_rewards_u = [simple_rewards[i] for i in urgency_tasks]
            
            print(f"{urgency_desc} ({len(urgency_tasks)} tasks):")
            print(f"  Simplified: {np.mean(simp_rewards_u):6.3f} ± {np.std(simp_rewards_u):5.3f}")
            print(f"  Simple:     {np.mean(simple_rewards_u):6.3f} ± {np.std(simple_rewards_u):5.3f}")
    
    # Show specific examples
    print()
    print("🔍 DETAILED EXAMPLES")
    print("=" * 50)
    
    example_indices = [0, 10, 20, 30, 40]  # Show a few examples
    print(f"{'Task':<6} | {'Priority':<8} | {'Days Left':<9} | {'Est Time':<8} | {'Simplified':<10} | {'Simple':<10} | {'Diff':<10}")
    print("-" * 80)
    
    for i in example_indices:
        if i < len(tasks):
            task = tasks[i]
            days_left = (task['deadline'] - current_date).days
            simp_r = simplified_rewards[i]
            simple_r = simple_rewards[i]
            diff = simp_r - simple_r
            
            print(f"{i+1:<6} | {task['priority']:<8} | {days_left:<9} | {task['estimated_time']:<8.1f} | {simp_r:<10.3f} | {simple_r:<10.3f} | {diff:<10.3f}")
    
    return simplified_rewards, simple_rewards

def estimate_episode_rewards(rewards, episode_length=1000):
    """Estimate what episode rewards would look like"""
    print()
    print("🎯 EPISODE REWARD ESTIMATION")
    print("=" * 50)
    
    # Simulate episode rewards
    n_episodes = 10
    episode_rewards = []
    
    for episode in range(n_episodes):
        # Sample rewards for this episode
        episode_reward = np.sum(np.random.choice(rewards, size=episode_length))
        episode_rewards.append(episode_reward)
    
    print(f"Estimated episode rewards ({episode_length} tasks per episode):")
    print(f"  Mean episode reward: {np.mean(episode_rewards):8.1f}")
    print(f"  Std episode reward:  {np.std(episode_rewards):8.1f}")
    print(f"  Min episode reward:  {np.min(episode_rewards):8.1f}")
    print(f"  Max episode reward:  {np.max(episode_rewards):8.1f}")
    
    if np.mean(episode_rewards) > 0:
        print("  ✅ POSITIVE episode rewards expected!")
    else:
        print("  ❌ NEGATIVE episode rewards expected!")
    
    return episode_rewards

def main():
    """Run comprehensive reward function analysis"""
    print("🚀 REWARD FUNCTION COMPARISON ANALYSIS")
    print("=" * 60)
    print()
    
    # Run comparison
    simplified_rewards, simple_rewards = compare_reward_functions()
    
    # Estimate episode performance
    print()
    print("SIMPLIFIED REWARD FUNCTION:")
    estimate_episode_rewards(simplified_rewards)
    
    print()
    print("SIMPLE REWARD FUNCTION:")
    estimate_episode_rewards(simple_rewards)
    
    # Final recommendation
    print()
    print("🎯 RECOMMENDATION")
    print("=" * 50)
    
    simp_mean = np.mean(simplified_rewards)
    simple_mean = np.mean(simple_rewards)
    
    if simp_mean > simple_mean:
        print("✅ The SIMPLIFIED reward function (reward_function.py) is performing better!")
        print(f"   Higher mean reward: {simp_mean:.3f} vs {simple_mean:.3f}")
    elif simple_mean > simp_mean:
        print("✅ The SIMPLE reward function (simple_reward_function.py) is performing better!")
        print(f"   Higher mean reward: {simple_mean:.3f} vs {simp_mean:.3f}")
    else:
        print("📊 Both reward functions are performing similarly.")
    
    if simp_mean > 0:
        print("✅ The simplified reward function should resolve negative reward issues!")
    else:
        print("⚠️  The simplified reward function may still have negative reward bias.")

if __name__ == "__main__":
    main()
