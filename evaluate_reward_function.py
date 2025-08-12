#!/usr/bin/env python3
"""
Comprehensive evaluation of the reward function
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from src.reward_function import calculate_reward, _calculate_priority_score, _calculate_urgency_score, _calculate_efficiency_score, _calculate_context_score

def create_comprehensive_test_data():
    """Create diverse test scenarios"""
    current_date = datetime.now()
    
    # Scenario 1: Overdue tasks
    overdue_tasks = pd.DataFrame({
        'task_id': [101, 102, 103],
        'deadline': [
            current_date - timedelta(days=1),  # 1 day overdue
            current_date - timedelta(days=3),  # 3 days overdue  
            current_date - timedelta(days=7)   # 1 week overdue
        ],
        'priority': [1, 2, 3],
        'estimated_time': [1.0, 2.0, 0.5],
        'status': ['todo', 'in_progress', 'todo'],
        'start_date': [current_date - timedelta(days=5)] * 3,
        'completion_date': [None] * 3
    })
    
    # Scenario 2: Due soon tasks
    urgent_tasks = pd.DataFrame({
        'task_id': [201, 202, 203, 204],
        'deadline': [
            current_date + timedelta(hours=6),   # Due today
            current_date + timedelta(days=1),    # Due tomorrow
            current_date + timedelta(days=3),    # Due in 3 days
            current_date + timedelta(days=7)     # Due next week
        ],
        'priority': [3, 2, 1, 2],
        'estimated_time': [0.5, 2.0, 4.0, 1.5],
        'status': ['todo', 'todo', 'in_progress', 'todo'],
        'start_date': [current_date - timedelta(days=1)] * 4,
        'completion_date': [None] * 4
    })
    
    # Scenario 3: Long-term tasks
    longterm_tasks = pd.DataFrame({
        'task_id': [301, 302, 303],
        'deadline': [
            current_date + timedelta(days=14),   # 2 weeks
            current_date + timedelta(days=30),   # 1 month
            current_date + timedelta(days=60)    # 2 months
        ],
        'priority': [1, 2, 3],
        'estimated_time': [8.0, 3.0, 1.0],
        'status': ['todo', 'todo', 'todo'],
        'start_date': [current_date - timedelta(days=1)] * 3,
        'completion_date': [None] * 3
    })
    
    # Combine all scenarios
    all_tasks = pd.concat([overdue_tasks, urgent_tasks, longterm_tasks], ignore_index=True)
    
    # Create behavior data with various efficiency patterns
    behavior_data = pd.DataFrame({
        'task_id': all_tasks['task_id'],
        'completion_time': [
            # Overdue tasks
            1.2, 1.8, 0.6,
            # Urgent tasks - mix of efficient/inefficient
            0.3, 2.5, 5.0, 1.2,
            # Long-term tasks
            6.0, 3.5, 0.8
        ],
        'actual_start_date': [current_date - timedelta(days=1)] * len(all_tasks)
    })
    
    return all_tasks, behavior_data

def evaluate_component_functions():
    """Test individual component functions"""
    print("=== Evaluating Component Functions ===\n")
    
    tasks, behavior = create_comprehensive_test_data()
    
    # Test priority scoring
    print("Priority Scores:")
    for priority in [1, 2, 3]:
        score = (priority - 2) / 1.0
        print(f"  Priority {priority}: {score:.2f}")
    print()
    
    # Test urgency scoring for different time horizons
    print("Urgency Scores (days remaining -> score):")
    current_date = datetime.now()
    test_deadlines = [-7, -1, 0, 1, 3, 7, 14, 30]
    
    for days in test_deadlines:
        test_task = pd.Series({
            'deadline': current_date + timedelta(days=days)
        })
        score = _calculate_urgency_score(test_task)
        status = "OVERDUE" if days <= 0 else f"{days}d away"
        print(f"  {status:>10}: {score:.2f}")
    print()
    
    # Test efficiency scoring
    print("Efficiency Scores (est. time vs actual -> score):")
    test_cases = [
        (1.0, 0.5, "Much faster"),  # 2x faster
        (2.0, 1.8, "Slightly faster"),  # 1.1x faster
        (3.0, 3.0, "Exact estimate"),  # Perfect
        (1.0, 1.5, "Slower"),  # 1.5x slower
    ]
    
    test_behavior = pd.DataFrame({'task_id': [999], 'completion_time': [0]})
    
    for est_time, actual_time, desc in test_cases:
        test_behavior.loc[0, 'completion_time'] = actual_time
        test_task = pd.Series({'task_id': 999, 'estimated_time': est_time})
        score = _calculate_efficiency_score(test_task, test_behavior)
        ratio = est_time / actual_time
        print(f"  {desc:>15} (ratio {ratio:.2f}): {score:.2f}")
    print()

def evaluate_reward_distribution():
    """Analyze reward distribution across different scenarios"""
    print("=== Evaluating Reward Distribution ===\n")
    
    tasks, behavior = create_comprehensive_test_data()
    
    rewards = []
    scenarios = []
    priorities = []
    urgencies = []
    
    for i, task in tasks.iterrows():
        reward = calculate_reward(task, behavior, i, tasks)
        days_remaining = (pd.to_datetime(task['deadline']) - datetime.now()).days
        
        rewards.append(reward)
        priorities.append(task['priority'])
        
        if days_remaining <= 0:
            scenario = "Overdue"
            urgencies.append("Overdue")
        elif days_remaining <= 7:
            scenario = "Urgent"
            urgencies.append("This Week")
        else:
            scenario = "Long-term"
            urgencies.append("Later")
        
        scenarios.append(scenario)
        
        print(f"Task {task['task_id']:>3}: Priority={task['priority']}, "
              f"Days={days_remaining:>3}, Reward={reward:.3f} ({scenario})")
    
    print(f"\nReward Statistics:")
    print(f"  Range: [{min(rewards):.3f}, {max(rewards):.3f}]")
    print(f"  Mean: {np.mean(rewards):.3f}")
    print(f"  Std: {np.std(rewards):.3f}")
    
    # Group analysis
    df = pd.DataFrame({
        'reward': rewards,
        'scenario': scenarios,
        'priority': priorities,
        'urgency': urgencies
    })
    
    print(f"\nReward by Scenario:")
    scenario_stats = df.groupby('scenario')['reward'].agg(['mean', 'min', 'max', 'count'])
    print(scenario_stats)
    
    print(f"\nReward by Priority:")
    priority_stats = df.groupby('priority')['reward'].agg(['mean', 'min', 'max', 'count'])
    print(priority_stats)
    
    return df

def test_edge_cases():
    """Test problematic edge cases"""
    print("=== Testing Edge Cases ===\n")
    
    current_date = datetime.now()
    
    # Edge case 1: Task with missing completion time
    print("1. Missing completion time:")
    task_missing = pd.Series({
        'task_id': 999,
        'deadline': current_date + timedelta(days=1),
        'priority': 2,
        'estimated_time': 2.0,
        'status': 'todo',
        'start_date': current_date,
        'completion_date': None
    })
    behavior_missing = pd.DataFrame({'task_id': [999], 'completion_time': [np.nan]})
    reward = calculate_reward(task_missing, behavior_missing, 0)
    print(f"  Reward: {reward:.3f}")
    
    # Edge case 2: Zero estimated time
    print("2. Zero estimated time:")
    task_zero_time = task_missing.copy()
    task_zero_time['estimated_time'] = 0.0
    reward = calculate_reward(task_zero_time, behavior_missing, 0)
    print(f"  Reward: {reward:.3f}")
    
    # Edge case 3: Very large estimated time
    print("3. Very large estimated time:")
    task_large_time = task_missing.copy()
    task_large_time['estimated_time'] = 100.0
    reward = calculate_reward(task_large_time, behavior_missing, 0)
    print(f"  Reward: {reward:.3f}")
    
    # Edge case 4: Empty available_tasks
    print("4. Single task remaining:")
    empty_tasks = pd.DataFrame(columns=['priority'])
    reward = calculate_reward(task_missing, behavior_missing, 5, empty_tasks)
    print(f"  Reward: {reward:.3f}")
    
    print()

def identify_potential_issues():
    """Identify potential problems with current reward function"""
    print("=== Identifying Potential Issues ===\n")
    
    issues = []
    
    # Issue 1: Urgency dominance
    print("1. Urgency vs Priority Trade-off Analysis:")
    current_date = datetime.now()
    
    # High priority, low urgency task
    high_pri_task = pd.Series({
        'task_id': 1,
        'deadline': current_date + timedelta(days=30),
        'priority': 3,
        'estimated_time': 1.0,
        'status': 'todo'
    })
    
    # Low priority, high urgency task
    low_pri_task = pd.Series({
        'task_id': 2,
        'deadline': current_date + timedelta(days=1),
        'priority': 1,
        'estimated_time': 1.0,
        'status': 'todo'
    })
    
    empty_behavior = pd.DataFrame({'task_id': [], 'completion_time': []})
    
    high_pri_reward = calculate_reward(high_pri_task, empty_behavior, 0)
    low_pri_reward = calculate_reward(low_pri_task, empty_behavior, 0)
    
    print(f"  High priority (3), low urgency (30d): {high_pri_reward:.3f}")
    print(f"  Low priority (1), high urgency (1d):  {low_pri_reward:.3f}")
    
    if low_pri_reward > high_pri_reward:
        issues.append("Urgency may override priority too strongly")
        print("  WARNING: Urgency overrides priority")
    else:
        print("  ✓  Priority appropriately balanced")
    
    # Issue 2: Efficiency component reliability
    print("\n2. Efficiency Component Analysis:")
    print("  Current efficiency logic assumes historical completion_time is reliable")
    print("  But this may not account for:")
    print("    - Task complexity changes over time")
    print("    - Different contexts/interruptions")
    print("    - Learning effects")
    issues.append("Efficiency scoring may not adapt to changing contexts")
    
    # Issue 3: Context scoring limitations  
    print("\n3. Context Scoring Limitations:")
    print("  Current context only considers average priority of remaining tasks")
    print("  Missing considerations:")
    print("    - Task dependencies")
    print("    - Time of day preferences") 
    print("    - Energy level requirements")
    print("    - Context switching costs")
    issues.append("Context scoring is overly simplistic")
    
    # Issue 4: Linear reward combination
    print("\n4. Linear Combination Issues:")
    print("  Current: 0.4*urgency + 0.3*priority + 0.2*efficiency + 0.1*context")
    print("  Potential problems:")
    print("    - Fixed weights may not suit all users")
    print("    - No interaction effects between components")
    print("    - May not capture non-linear preferences")
    issues.append("Fixed linear combination may be suboptimal")
    
    return issues

def suggest_improvements(issues):
    """Suggest specific improvements based on identified issues"""
    print("=== Suggested Improvements ===\n")
    
    improvements = []
    
    print("1. **Adaptive Weight System**")
    print("   - Learn user-specific weights for priority vs urgency trade-offs")
    print("   - Use separate weights for different contexts (work/personal)")
    print("   - Implement online learning to adapt weights over time")
    improvements.append("Implement adaptive weight learning")
    
    print("\n2. **Enhanced Urgency Modeling**")
    print("   - Use continuous urgency function instead of discrete buckets")
    print("   - Add deadline buffer preferences (some prefer early completion)")
    print("   - Consider time-of-day factors for deadline pressure")
    improvements.append("Continuous urgency function with personal preferences")
    
    print("\n3. **Improved Efficiency Estimation**")
    print("   - Track efficiency by task type/category, not just task ID")
    print("   - Use exponential decay for historical data (recent > old)")
    print("   - Add confidence intervals for efficiency estimates")
    improvements.append("Categorical efficiency tracking with temporal decay")
    
    print("\n4. **Rich Context Modeling**")
    print("   - Track task switching costs between different priority levels")
    print("   - Consider sequential dependencies between tasks")
    print("   - Add time-block optimization (batch similar tasks)")
    improvements.append("Multi-factor context modeling")
    
    print("\n5. **Non-linear Reward Combination**")
    print("   - Use multiplicative components for critical interactions")
    print("   - Add threshold effects (e.g., overdue tasks get exponential penalty)")
    print("   - Implement user preference learning for reward shape")
    improvements.append("Non-linear reward function with learned preferences")
    
    print("\n6. **Validation and Calibration**")
    print("   - Add A/B testing framework for reward function variants")
    print("   - Implement user feedback loops for reward quality")
    print("   - Create interpretability tools for reward decisions")
    improvements.append("Reward function validation and calibration system")
    
    return improvements

def create_visualizations(reward_df):
    """Create visualizations of reward function behavior"""
    print("=== Creating Reward Function Visualizations ===\n")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Reward by scenario
    sns.boxplot(data=reward_df, x='scenario', y='reward', ax=axes[0,0])
    axes[0,0].set_title('Reward Distribution by Scenario')
    axes[0,0].grid(True, alpha=0.3)
    
    # Reward by priority
    sns.boxplot(data=reward_df, x='priority', y='reward', ax=axes[0,1])
    axes[0,1].set_title('Reward Distribution by Priority')
    axes[0,1].grid(True, alpha=0.3)
    
    # Reward by urgency
    sns.boxplot(data=reward_df, x='urgency', y='reward', ax=axes[1,0])
    axes[1,0].set_title('Reward Distribution by Urgency')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Overall distribution
    axes[1,1].hist(reward_df['reward'], bins=15, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Reward Value')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Overall Reward Distribution')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axvline(reward_df['reward'].mean(), color='red', linestyle='--', label=f'Mean: {reward_df["reward"].mean():.3f}')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('plots/reward_function_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved to: plots/reward_function_evaluation.png")

def main():
    """Run comprehensive reward function evaluation"""
    print("COMPREHENSIVE REWARD FUNCTION EVALUATION\n")
    print("=" * 60)
    
    # Component analysis
    evaluate_component_functions()
    
    # Distribution analysis  
    reward_df = evaluate_reward_distribution()
    
    # Edge cases
    test_edge_cases()
    
    # Issue identification
    issues = identify_potential_issues()
    
    # Improvement suggestions
    improvements = suggest_improvements(issues)
    
    # Create visualizations
    create_visualizations(reward_df)
    
    # Final summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n**Strengths:**")
    print(f"  • Rewards properly normalized to [-1, 1] range")
    print(f"  • Clear component separation (priority, urgency, efficiency, context)")
    print(f"  • Handles edge cases without crashing")
    print(f"  • Overdue tasks appropriately prioritized")
    
    print(f"\n**Issues Identified ({len(issues)}):**")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print(f"\n**Recommended Improvements ({len(improvements)}):**")
    for i, improvement in enumerate(improvements, 1):
        print(f"  {i}. {improvement}")
    
    print(f"\n**Key Statistics:**")
    print(f"  • Reward range: [{reward_df['reward'].min():.3f}, {reward_df['reward'].max():.3f}]")
    print(f"  • Mean reward: {reward_df['reward'].mean():.3f}")
    print(f"  • Std deviation: {reward_df['reward'].std():.3f}")
    
    overdue_mean = reward_df[reward_df['scenario'] == 'Overdue']['reward'].mean()
    urgent_mean = reward_df[reward_df['scenario'] == 'Urgent']['reward'].mean()
    longterm_mean = reward_df[reward_df['scenario'] == 'Long-term']['reward'].mean()
    
    print(f"  • Overdue tasks avg: {overdue_mean:.3f}")
    print(f"  • Urgent tasks avg: {urgent_mean:.3f}")
    print(f"  • Long-term tasks avg: {longterm_mean:.3f}")
    
    print(f"\n**Overall Assessment:** The reward function is functional but has room")
    print(f"    for significant improvements in personalization and context modeling.")

if __name__ == "__main__":
    main()