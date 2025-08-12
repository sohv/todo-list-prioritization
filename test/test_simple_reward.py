#!/usr/bin/env python3
"""
Test script for the simple reward function.
Demonstrates basic functionality and gradual complexity introduction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simple_reward_function import (
    calculate_simple_reward,
    calculate_progressive_reward,
    compare_tasks,
    get_best_task
)

def create_simple_task(task_id=1, priority=2, deadline_days=7, status='pending', estimated_time=2.0):
    """Create a simple test task"""
    return {
        'task_id': task_id,
        'priority': priority,
        'deadline': (datetime.now() + timedelta(days=deadline_days)).strftime('%Y-%m-%d'),
        'status': status,
        'estimated_time': estimated_time
    }

def test_basic_functionality():
    """Test basic reward calculation"""
    print("🧪 Testing Basic Functionality...")
    
    # Create test tasks
    high_urgent = create_simple_task(priority=3, deadline_days=0)  # Due today
    medium_normal = create_simple_task(priority=2, deadline_days=7)  # Due next week
    low_distant = create_simple_task(priority=1, deadline_days=30)  # Due next month
    
    # Calculate rewards
    reward_high = calculate_simple_reward(high_urgent)
    reward_medium = calculate_simple_reward(medium_normal)
    reward_low = calculate_simple_reward(low_distant)
    
    print(f"   High priority, due today: {reward_high:.3f}")
    print(f"   Medium priority, due next week: {reward_medium:.3f}")
    print(f"   Low priority, due next month: {reward_low:.3f}")
    
    # Verify ordering
    assert reward_high > reward_medium > reward_low, "Reward ordering incorrect"
    print("   Basic functionality test passed!")

def test_task_comparison():
    """Test task comparison and ranking"""
    print("\n🧪 Testing Task Comparison...")
    
    tasks = [
        create_simple_task(1, priority=1, deadline_days=30),  # Low priority, distant
        create_simple_task(2, priority=3, deadline_days=1),   # High priority, urgent
        create_simple_task(3, priority=2, deadline_days=7),   # Medium priority, normal
        create_simple_task(4, priority=3, deadline_days=14),  # High priority, not urgent
    ]
    
    # Compare tasks
    ranked_tasks = compare_tasks(tasks)
    
    print("   Task ranking (highest to lowest reward):")
    for i, (task, reward) in enumerate(ranked_tasks, 1):
        print(f"   {i}. Task {task['task_id']} (P{task['priority']}, {task['deadline']}): {reward:.3f}")
    
    # Get best task
    best_task, best_reward = get_best_task(tasks)
    print(f"\n   Best task: Task {best_task['task_id']} with score {best_reward:.3f}")
    
    # Verify high priority urgent task wins
    assert best_task['task_id'] == 2, "Expected task 2 (high priority, urgent) to win"
    print("   Task comparison test passed!")

def test_progressive_complexity():
    """Test gradual complexity introduction"""
    print("\n🧪 Testing Progressive Complexity...")
    
    # Create test task
    task = create_simple_task(priority=2, deadline_days=-3, status='in_progress')  # Overdue, in progress
    
    # Test different complexity levels
    basic_reward = calculate_simple_reward(task)
    status_reward = calculate_progressive_reward(task, include_status=True)
    full_reward = calculate_progressive_reward(task, include_status=True, include_penalties=True)
    
    print(f"   Basic reward: {basic_reward:.3f}")
    print(f"   With status bonus: {status_reward:.3f}")
    print(f"   With status + penalties: {full_reward:.3f}")
    
    # Status bonus should increase reward, overdue penalty should decrease it
    assert status_reward > basic_reward, "Status bonus should increase reward"
    assert full_reward < status_reward, "Overdue penalty should decrease reward"
    
    print("   Progressive complexity test passed!")

def test_with_user_behavior():
    """Test with historical user behavior data"""
    print("\n🧪 Testing with User Behavior Data...")
    
    # Create user behavior data
    user_behavior = pd.DataFrame([
        {'task_id': 1, 'completion_time': 1.5},  # Fast completion
        {'task_id': 2, 'completion_time': 4.0},  # Slow completion
    ])
    
    # Create tasks with different efficiency profiles
    fast_task = create_simple_task(1, estimated_time=2.0)  # Should be efficient
    slow_task = create_simple_task(2, estimated_time=2.0)  # Should be inefficient
    new_task = create_simple_task(3, estimated_time=1.0)   # No history, short
    
    reward_fast = calculate_simple_reward(fast_task, user_behavior)
    reward_slow = calculate_simple_reward(slow_task, user_behavior)
    reward_new = calculate_simple_reward(new_task, user_behavior)
    
    print(f"   Fast completion task: {reward_fast:.3f}")
    print(f"   Slow completion task: {reward_slow:.3f}")
    print(f"   New short task: {reward_new:.3f}")
    
    # Fast task should have higher reward than slow task
    assert reward_fast > reward_slow, "Fast task should have higher reward"
    
    print("   User behavior test passed!")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🧪 Testing Edge Cases...")
    
    # Task with missing deadline
    task_no_deadline = {'task_id': 1, 'priority': 2}
    reward_no_deadline = calculate_simple_reward(task_no_deadline)
    print(f"   Task without deadline: {reward_no_deadline:.3f}")
    
    # Task with invalid deadline
    task_bad_deadline = {'task_id': 2, 'priority': 2, 'deadline': 'invalid-date'}
    reward_bad_deadline = calculate_simple_reward(task_bad_deadline)
    print(f"   Task with bad deadline: {reward_bad_deadline:.3f}")
    
    # Empty task list
    best_task, best_reward = get_best_task([])
    assert best_task is None, "Empty task list should return None"
    print(f"   Empty task list: {best_task}, {best_reward}")
    
    # All rewards should be within bounds
    assert -1.0 <= reward_no_deadline <= 1.0, "Reward out of bounds"
    assert -1.0 <= reward_bad_deadline <= 1.0, "Reward out of bounds"
    
    print("   Edge cases test passed!")

def demo_migration_path():
    """Demonstrate migration path from simple to complex"""
    print("\nMigration Path Demo:")
    print("=" * 40)
    
    tasks = [
        create_simple_task(1, priority=3, deadline_days=0, status='pending'),
        create_simple_task(2, priority=2, deadline_days=-5, status='in_progress'),  # Overdue
        create_simple_task(3, priority=1, deadline_days=14, status='blocked'),
    ]
    
    print("Phase 1: Basic reward function")
    basic_ranked = compare_tasks(tasks)
    for task, reward in basic_ranked:
        print(f"   Task {task['task_id']}: {reward:.3f}")
    
    print("\nPhase 2: Add status awareness")
    status_ranked = [(task, calculate_progressive_reward(task, include_status=True)) for task, _ in basic_ranked]
    status_ranked.sort(key=lambda x: x[1], reverse=True)
    for task, reward in status_ranked:
        print(f"   Task {task['task_id']}: {reward:.3f}")
    
    print("\nPhase 3: Add penalty patterns")
    full_ranked = [(task, calculate_progressive_reward(task, include_status=True, include_penalties=True)) for task, _ in basic_ranked]
    full_ranked.sort(key=lambda x: x[1], reverse=True)
    for task, reward in full_ranked:
        print(f"   Task {task['task_id']}: {reward:.3f}")
    
    print("\nNotice how task rankings change as complexity is added!")
    print("This allows gradual introduction based on learning performance.")

def main():
    """Run all tests and demonstrations"""
    print("Simple Reward Function Testing")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_task_comparison()
        test_progressive_complexity()
        test_with_user_behavior()
        test_edge_cases()
        
        demo_migration_path()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("\nSimple Reward Function Features:")
        print("   3-component basic model (priority, urgency, efficiency)")
        print("   Task comparison and ranking utilities")
        print("   Progressive complexity introduction")
        print("   Robust error handling")
        print("   Clear migration path to advanced features")
        print("\nRecommended Usage:")
        print("   Start with basic reward function for prototyping")
        print("   Add status awareness when task switching becomes important")
        print("   Add penalties when pattern recognition is needed")
        print("   Migrate to full advanced function for production RL systems")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())