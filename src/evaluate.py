import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from environment import TodoListEnv
from dqn_agent import DQNAgent
from utils import load_data

def evaluate_model(
    model_path="models/dqn_model_final.keras",
    test_data_dir="data/test",
    num_episodes=5,
    max_steps_per_episode=20
):
    """
    Evaluate the trained model on test data with status-aware evaluation.
    """
    print("Starting evaluation on test data...")
    
    try:
        # Load test data
        print("Loading test data...")
        test_tasks = pd.read_csv(os.path.join(test_data_dir, 'tasks.csv'))
        test_behavior = pd.read_csv(os.path.join(test_data_dir, 'user_behavior.csv'))
        print(f"Loaded {len(test_tasks)} test tasks")
        
        # Print test data statistics
        print("\nTest Data Statistics:")
        print("\nTask Status Distribution:")
        print(test_tasks['status'].value_counts(normalize=True))
        print("\nPriority Distribution:")
        print(test_tasks['priority'].value_counts())
        print("\nDeadline Range:")
        print(f"Earliest: {test_tasks['deadline'].min()}")
        print(f"Latest: {test_tasks['deadline'].max()}")
        
        # Initialize environment with test data
        print("\nInitializing environment...")
        env = TodoListEnv(test_tasks, test_behavior)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(f"Environment initialized with state size: {state_size}, action size: {action_size}")
        
        # Initialize agent
        print("\nInitializing agent...")
        agent = DQNAgent(state_size, action_size)
        
        # Load trained model
        print(f"Loading model from {model_path}")
        try:
            agent.model = tf.keras.models.load_model(model_path, compile=False)
            agent.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=agent.learning_rate),
                loss='huber'
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Evaluation metrics
        episode_rewards = []
        episode_lengths = []
        action_frequencies = np.zeros(action_size)
        status_based_rewards = {
            'completed': [],
            'in_progress': [],
            'todo': []
        }
        
        # Run evaluation episodes
        print("\nStarting evaluation episodes...")
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            try:
                state = env.reset()
                state = np.reshape(state, [1, state_size])
                total_reward = 0
                steps = 0
                episode_actions = []
                episode_status_rewards = {
                    'completed': 0,
                    'in_progress': 0,
                    'todo': 0
                }
                
                while steps < max_steps_per_episode:
                    # Get and validate action
                    action = agent.act(state, training=False)
                    if action >= action_size:
                        print(f"  Warning: Invalid action {action}, clipping to valid range")
                        action = action % action_size
                    
                    # Track action
                    action_frequencies[action] += 1
                    episode_actions.append(action)
                    
                    # Get task status for the chosen action
                    task_status = test_tasks.iloc[action]['status']
                    
                    # Take step
                    next_state, reward, done, _ = env.step(action)
                    
                    # Track reward by status
                    episode_status_rewards[task_status] += reward
                    
                    # Print step information
                    print(f"  Step {steps + 1}:")
                    print(f"    Action: {action}")
                    print(f"    Task Status: {task_status}")
                    print(f"    Reward: {reward:.2f}")
                    print(f"    Done: {done}")
                    
                    next_state = np.reshape(next_state, [1, state_size])
                    total_reward += reward
                    state = next_state
                    steps += 1
                    
                    if done:
                        print(f"  Episode naturally finished after {steps} steps")
                        break
                
                if steps >= max_steps_per_episode:
                    print(f"  Warning: Episode hit max steps ({max_steps_per_episode})")
                
                # Store episode metrics
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                for status, reward in episode_status_rewards.items():
                    status_based_rewards[status].append(reward)
                
                # Episode summary
                print(f"\n  Episode {episode + 1} Summary:")
                print(f"    Steps: {steps}")
                print(f"    Total Reward: {total_reward:.2f}")
                print("    Rewards by Status:")
                for status, reward in episode_status_rewards.items():
                    print(f"      {status}: {reward:.2f}")
                print(f"    Unique actions taken: {len(set(episode_actions))}")
                print("  ------------------------")
                
            except Exception as e:
                print(f"Error during episode {episode + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Detailed evaluation summary
        if episode_rewards:
            print("\nDetailed Evaluation Summary:")
            
            print("\nOverall Performance:")
            print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"Best Episode Reward: {max(episode_rewards):.2f}")
            print(f"Worst Episode Reward: {min(episode_rewards):.2f}")
            
            print("\nPerformance by Task Status:")
            for status in status_based_rewards:
                if status_based_rewards[status]:
                    mean_reward = np.mean(status_based_rewards[status])
                    std_reward = np.std(status_based_rewards[status])
                    print(f"{status.title()} Tasks:")
                    print(f"  Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            print("\nEpisode Statistics:")
            print(f"Average Length: {np.mean(episode_lengths):.1f} steps")
            print(f"Max Length: {max(episode_lengths)} steps")
            print(f"Min Length: {min(episode_lengths)} steps")
            
            print("\nAction Distribution:")
            for action in range(action_size):
                frequency = action_frequencies[action]
                if frequency > 0:
                    percentage = (frequency / action_frequencies.sum()) * 100
                    task_status = test_tasks.iloc[action]['status']
                    print(f"Action {action} ({task_status}): {frequency:.0f} times ({percentage:.1f}%)")
        else:
            print("\nNo episodes completed successfully")
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_model(
        model_path="models/dqn_model_final.keras",
        test_data_dir="data/test",
        num_episodes=5,
        max_steps_per_episode=20
    )