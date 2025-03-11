import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os

# Define the neural network model using PyTorch
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(state_size),
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, x):
        return self.model(x)

class DQNAgent:
    class _ModelWrapper:
        """Wrapper class to mimic TensorFlow model API"""
        def __init__(self, agent_instance):
            self.agent = agent_instance
        
        def predict(self, state, verbose=0):
            """Mimic TF model.predict() for compatibility"""
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.agent.device)
                self.agent.dqn_model.eval()
                predictions = self.agent.dqn_model(state_tensor).cpu().numpy()
                self.agent.dqn_model.train()
                return predictions
        
        def save(self, filepath):
            """Mimic TF model.save() for compatibility"""
            self.agent.save(filepath)
        
        def compile(self, **kwargs):
            """Dummy method for compatibility"""
            pass

    def __init__(self, state_size, action_size, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.priority_memory = deque(maxlen=1000)  # Special buffer for high-reward experiences
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.0005
        self.tau = 0.001  # Soft update parameter
        
        # Set device (CPU or MPS)
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"DQNAgent using device: {self.device}")
        
        # Create PyTorch models and move to device
        self.dqn_model = DQNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.dqn_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.HuberLoss()  # Huber loss (similar to TF's 'huber')
        
        # Create model wrapper for TF compatibility
        self._model_wrapper_instance = self._ModelWrapper(self)
        
        # Initialize target model with same weights
        self.update_target_model()
    
    @property
    def model(self):
        """Property to provide TensorFlow-like model access"""
        return self._model_wrapper_instance
    
    @model.setter
    def model(self, model):
        """Handle attempts to set the model directly"""
        print("Warning: Attempted to set model directly. Using PyTorch model instead.")
    
    def update_target_model(self):
        """Soft update target model parameters."""
        for target_param, param in zip(self.target_model.parameters(), self.dqn_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        
        # Store high-reward experiences in priority buffer
        if len(self.memory) > 1:
            recent_rewards = [x[2] for x in list(self.memory)[-100:]]
            if reward > np.mean(recent_rewards):
                self.priority_memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action based on epsilon-greedy policy."""
        if training and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        # Convert state to PyTorch tensor and move to device
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Get action values
        self.dqn_model.eval()
        with torch.no_grad():
            act_values = self.dqn_model(state_tensor)
        self.dqn_model.train()
        
        # Return best action
        return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        # Mix regular and priority experiences
        n_priority = batch_size // 4  # 25% priority experiences
        n_regular = batch_size - n_priority
        
        minibatch = random.sample(self.memory, n_regular)
        if self.priority_memory and n_priority > 0:
            minibatch.extend(random.sample(self.priority_memory, 
                                         min(n_priority, len(self.priority_memory))))
        
        # Prepare batch data
        states = torch.FloatTensor(np.array([i[0][0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3][0] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)
        
        # Double DQN update
        self.dqn_model.eval()
        with torch.no_grad():
            next_actions = torch.argmax(self.dqn_model(next_states), dim=1)
            next_q_values = self.target_model(next_states)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        self.dqn_model.train()
        
        # Current Q values
        current_q_values = self.dqn_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss and update model
        loss = self.criterion(current_q_values, target_q_values)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.dqn_model.parameters(), 1.0)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Soft update target network
        self.update_target_model()
        
        return loss.item()
    
    def save(self, filepath):
        """Save model to a file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Ensure the file has the correct extension
        if not (filepath.endswith('.pt') or filepath.endswith('.pth')):
            filepath = f"{filepath}.pt"
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.dqn_model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from a file."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.dqn_model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {filepath}")
        else:
            print(f"Error: Model file {filepath} not found")
    
    # For compatibility with TensorFlow code
    def model_save(self, filepath):
        """Compatibility method for TF's model.save()"""
        # Convert .keras extension to .pt if present
        if filepath.endswith('.keras'):
            filepath = filepath[:-6] + '.pt'
        self.save(filepath)