#!/usr/bin/env python3
"""
Standalone A2C training script for task prioritization
"""

import sys
import os
sys.path.append('/root/todo')

from src.train_actor_critic import train_a2c_model

def main():
    """Train only A2C agent with optimized parameters"""
    print("=== A2C Training for Task Prioritization ===")
    print("Training with parameters optimized for comparison with DQN...\n")
    
    # Train A2C with optimized parameters
    a2c_agent = train_a2c_model(
        train_data_dir="data/train",
        test_data_dir="data/test",
        episodes=1000,                    # Match DQN episodes
        max_steps_per_episode=300,        # Match DQN tasks per episode  
        update_frequency=20,              # Reasonable for A2C
        save_interval=100,               # Save every 100 episodes
        plot_interval=50,                # Plot every 50 episodes
        validation_interval=200,         # Validate every 200 episodes
        early_stopping_patience=500      # Early stopping patience
    )
    
    print("\n=== A2C Training Complete ===")
    print("Results saved to:")
    print("  - Models: models/a2c_model_final.pt")
    print("  - Plots: plots/a2c/")
    print("  - Logs: logs/a2c_training_log.txt")
    print("\nReady for comparison with DQN results!")

if __name__ == "__main__":
    main()
