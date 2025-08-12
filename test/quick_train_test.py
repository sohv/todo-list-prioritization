#!/usr/bin/env python3
"""
Quick training test with the fixed implementation
"""

from src.train import train_model

if __name__ == "__main__":
    # Run a very short training session to validate everything works
    print("Running quick training test with fixes...")
    train_model(
        train_data_dir="data/train",
        test_data_dir="data/test", 
        episodes=20,  # Very short test
        batch_size=32,
        save_interval=10,
        plot_interval=10,
        validation_interval=10,
        early_stopping_patience=50
    )
    print("Quick training test completed successfully!")