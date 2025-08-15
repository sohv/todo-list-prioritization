#!/usr/bin/env python3
"""
Standalone A3C training script for task prioritization
"""

import sys
import os
sys.path.append('/root/todo')

from src.train_actor_critic import train_a3c_model

def main():
    """Train only A3C agent with optimized parameters"""
    print("=== A3C Training for Task Prioritization ===")
    print("Training with parameters optimized for comparison with A2C and DQN...\n")
    
    # Train A3C with optimized parameters
    a3c_agent = train_a3c_model(
        train_data_dir="data/train",
        test_data_dir="data/test",
        num_workers=4,                    # 4 parallel workers
        episodes_per_worker=250,          # 250 episodes per worker = 1000 total episodes
        max_steps_per_episode=300,        # Match A2C/DQN tasks per episode  
        update_frequency=20               # Reasonable for A3C
    )
    
    print("\n=== A3C Training Complete ===")
    print("Results saved to:")
    print("  - Models: models/a3c_model_final.pt")
    print("  - Plots: plots/a3c/")
    print("  - Logs: logs/a3c_training_log.txt")
    print("\nReady for comparison with A2C and DQN results!")

if __name__ == "__main__":
    main()
