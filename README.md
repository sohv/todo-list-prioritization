<!-- Project documentation -->
# Adaptive Task prioritization in To-Do List

A RL-based task prioritization system that helps optimize task scheduling based on deadlines, priorities, and estimated completion times.

## Project Structure

```
.
├── config/           
├── data/            
├── logs/            
├── models/          
├── plots/           
├── src/             
│   ├── environment.py    
│   ├── dqn_agent.py      
│   ├── evaluate.py       
│   ├── reward_function.py 
│   ├── train.py          
│   └── utils.py          
├── generate_data.py 
├── main.py          
└── requirements.txt 
```

## Features

- Reinforcement learning-based task prioritization
- Custom Gym environment for task management
- Deep Q-Network (DQN) implementation
- Synthetic task data generation
- Model evaluation and visualization tools

## Requirements

- Python 3.12
- Dependencies listed in `requirements.txt`:
  - numpy
  - pandas
  - gym
  - torch
  - matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sohv/todo-list-prioritization.git
cd todo-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Generate synthetic task data:
```bash
python generate_data.py
```

2. Train the agent:
```bash
python main.py
```

3. Evaluate the trained model:
```bash
python src/evaluate.py
```

## Project Components

### Environment
The `TodoListEnv` class in `environment.py` implements a custom Gym environment that:
- Manages task states (deadlines, priorities, estimated times)
- Provides action space for task selection
- Calculates rewards based on task completion

### Agent
The DQN agent in `dqn_agent.py` implements:
- Deep Q-Network architecture
- Experience replay buffer
- Target network for stable training

### Training
The training pipeline in `train.py` handles:
- Model training loop
- Experience collection
- Model checkpointing
- Performance logging

### Evaluation
The evaluation module in `evaluate.py` provides:
- Model performance assessment
- Task completion metrics
- Visualization tools

## Future Improvements

### Advanced RL Algorithms
The current DQN implementation can be enhanced by implementing:
- A3C (Asynchronous Advantage Actor-Critic) for improved parallel training and sample efficiency
- A2C (Advantage Actor-Critic) for more stable and simpler training process

### Further Work
- Multi-agent task scheduling
- Integration with calendar APIs

## License

This project is licensed under the terms specified in the MIT License.