import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
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
        def __init__(self, agent_instance):
            self.agent = agent_instance
        
        def predict(self, state, verbose=0):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.agent.device)
                self.agent.dqn_model.eval()
                predictions = self.agent.dqn_model(state_tensor).cpu().numpy()
                self.agent.dqn_model.train()
                return predictions
        
        def save(self, filepath):
            self.agent.save(filepath)
        
        def compile(self, **kwargs):
            pass

    def __init__(self, state_size, action_size, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  
        self.priority_memory = deque(maxlen=1000)  
        self.gamma = 0.99  
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.0005
        self.tau = 0.001  
        
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"DQNAgent using device: {self.device}")
        self.dqn_model = DQNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.dqn_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.HuberLoss()  
        self._model_wrapper_instance = self._ModelWrapper(self)
        self.update_target_model()
    
    @property
    def model(self):
        return self._model_wrapper_instance
    
    @model.setter
    def model(self, model):
        print("Warning: Attempted to set model directly. Using PyTorch model instead.")
    
    def update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.dqn_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) > 1:
            recent_rewards = [x[2] for x in list(self.memory)[-100:]]
            if reward > np.mean(recent_rewards):
                self.priority_memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        self.dqn_model.eval()
        with torch.no_grad():
            act_values = self.dqn_model(state_tensor)
        self.dqn_model.train()
        return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        n_priority = batch_size // 4  
        n_regular = batch_size - n_priority  
        minibatch = random.sample(self.memory, n_regular)
        if self.priority_memory and n_priority > 0:
            minibatch.extend(random.sample(self.priority_memory, 
                                         min(n_priority, len(self.priority_memory))))
        
        states = torch.FloatTensor(np.array([i[0][0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3][0] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)
        
        self.dqn_model.eval()
        with torch.no_grad():
            next_actions = torch.argmax(self.dqn_model(next_states), dim=1)
            next_q_values = self.target_model(next_states)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        self.dqn_model.train()
        
        current_q_values = self.dqn_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.criterion(current_q_values, target_q_values)
        
        torch.nn.utils.clip_grad_norm_(self.dqn_model.parameters(), 1.0)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.update_target_model()
        return loss.item()
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if not (filepath.endswith('.pt') or filepath.endswith('.pth')):
            filepath = f"{filepath}.pt"
        
        torch.save({
            'model_state_dict': self.dqn_model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.dqn_model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {filepath}")
        else:
            print(f"Error: Model file {filepath} not found")
    
    def model_save(self, filepath):
        if filepath.endswith('.keras'):
            filepath = filepath[:-6] + '.pt'
        self.save(filepath)