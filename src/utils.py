# utility functions (data preprocessing, reward function, etc.)

import pandas as pd

def load_data():
    tasks = pd.read_csv("data/tasks.csv")
    user_behavior = pd.read_csv("data/user_behavior.csv")
    return tasks, user_behavior