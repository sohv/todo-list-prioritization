import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import os

# Experience tuple for cleaner replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # More robust architecture with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        # Dueling DQN architecture
        features = self.feature_extractor(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        
    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        # New experiences get maximum priority
        max_priority = self.priorities.max() if self.buffer else 1.0
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None, None, None
            
        priorities = self.priorities[:len(self.buffer)]
        # Ensure priorities are positive and avoid NaN
        priorities = np.maximum(priorities, 1e-8)
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        
        # Handle edge case where sum is 0 or NaN
        if probs_sum == 0 or np.isnan(probs_sum):
            # Use uniform distribution as fallback
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        else:
            probs /= probs_sum
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights_max = weights.max()
        if weights_max == 0 or np.isnan(weights_max):
            weights = np.ones_like(weights)
        else:
            weights /= weights_max
        
        return experiences, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    class _ModelWrapper:
        def __init__(self, agent_instance):
            self.agent = agent_instance
        
        def predict(self, state, verbose=0):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.agent.device)
                if len(state_tensor.shape) == 1:
                    state_tensor = state_tensor.unsqueeze(0)
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
        
        # Improved replay buffer with prioritized experience replay
        self.memory = PrioritizedReplayBuffer(50000, alpha=0.6)
        
        # Hyperparameters
        self.gamma = 0.99  
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Slower decay for better exploration
        self.learning_rate = 0.0001  # Lower learning rate for stability
        self.tau = 0.005  # Faster target network updates
        self.update_frequency = 4  # Update every N steps
        self.target_update_frequency = 1000  # Hard update target network
        
        # Training metrics
        self.step_count = 0
        self.loss_history = deque(maxlen=1000)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"DQNAgent using device: {self.device}")
        
        # Double DQN setup
        self._create_networks()
        
        self._model_wrapper_instance = self._ModelWrapper(self)
    
    def _create_networks(self):
        """Create DQN networks"""
        self.dqn_model = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.AdamW(
            self.dqn_model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        self.criterion = nn.SmoothL1Loss()
        
        # Initialize target network
        self.hard_update_target_model()
    
    @property
    def model(self):
        return self._model_wrapper_instance
    
    @model.setter
    def model(self, model):
        print("Warning: Attempted to set model directly. Using PyTorch model instead.")
    
    def hard_update_target_model(self):
        """Hard update: copy weights from main network to target network"""
        self.target_model.load_state_dict(self.dqn_model.state_dict())
    
    def soft_update_target_model(self):
        """Soft update: slowly blend target network with main network"""
        for target_param, param in zip(self.target_model.parameters(), self.dqn_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def remember(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
    
    def act(self, state, valid_actions=None, training=True):
        """Select action using epsilon-greedy policy with action masking"""
        self.step_count += 1
        
        if training and np.random.rand() <= self.epsilon:
            # Random action from valid actions only
            if valid_actions is not None and len(valid_actions) > 0:
                # Use faster random selection for large action spaces
                random_idx = np.random.randint(0, len(valid_actions))
                return valid_actions[random_idx]
            return np.random.choice(self.action_size)
        
        # Get Q-values
        state_tensor = torch.FloatTensor(state).to(self.device)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        self.dqn_model.eval()
        with torch.no_grad():
            q_values = self.dqn_model(state_tensor)
        self.dqn_model.train()
        
        # Apply action masking if provided
        if valid_actions is not None and len(valid_actions) > 0:
            # For efficiency with large action spaces, directly index valid actions
            if len(valid_actions) > 1000:  # Use efficient approach for large action spaces
                valid_actions_tensor = torch.tensor(valid_actions, device=self.device)
                valid_q_values = q_values[0, valid_actions_tensor]
                best_valid_idx = torch.argmax(valid_q_values).item()
                return valid_actions[best_valid_idx]
            else:
                # Original masking approach for smaller action spaces
                masked_q_values = q_values.clone()
                valid_mask = torch.zeros(self.action_size, dtype=torch.bool, device=self.device)
                valid_mask[valid_actions] = True
                masked_q_values[0, ~valid_mask] = -float('inf')
                return torch.argmax(masked_q_values[0]).item()
        
        return torch.argmax(q_values[0]).item()
    
    def replay(self, batch_size):
        """Train the agent using prioritized experience replay and Double DQN"""
        # Only train if we have enough experience and at the right frequency
        if len(self.memory) < batch_size or self.step_count % self.update_frequency != 0:
            return None
            
        # Annealing beta for importance sampling
        beta = min(1.0, 0.4 + 0.6 * (self.step_count / 100000))
        
        # Sample from prioritized replay buffer
        experiences, indices, weights = self.memory.sample(batch_size, beta)
        if experiences is None:
            return None
            
        # Convert experiences to tensors (efficiently using numpy arrays first)
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.BoolTensor(np.array([e.done for e in experiences])).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.dqn_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            # Get actions from main network
            next_actions = self.dqn_model(next_states).argmax(dim=1)
            # Get Q-values from target network
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Compute target Q-values
            target_q_values = rewards + (~dones * self.gamma * next_q_values)
        
        # Compute TD errors for priority updates
        td_errors = torch.abs(current_q_values - target_q_values)
        
        # Weighted loss using importance sampling
        loss = (weights * self.criterion(current_q_values, target_q_values)).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn_model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        priorities = td_errors.detach().cpu().numpy() + 1e-6  # Small epsilon to avoid zero priorities
        self.memory.update_priorities(indices, priorities)
        
        # Soft update target network
        self.soft_update_target_model()
        
        # Periodic hard update
        if self.step_count % self.target_update_frequency == 0:
            self.hard_update_target_model()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Store loss for tracking
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if not (filepath.endswith('.pt') or filepath.endswith('.pth')):
            filepath = f"{filepath}.pt"
        
        torch.save({
            'model_state_dict': self.dqn_model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hyperparameters': {
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'learning_rate': self.learning_rate,
                'tau': self.tau
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # Check if dimensions match the saved model
            if 'state_size' in checkpoint and 'action_size' in checkpoint:
                saved_state_size = checkpoint['state_size']
                saved_action_size = checkpoint['action_size']
                
                if saved_state_size != self.state_size or saved_action_size != self.action_size:
                    print(f"Warning: DQN model dimension mismatch!")
                    print(f"  Saved: state_size={saved_state_size}, action_size={saved_action_size}")
                    print(f"  Current: state_size={self.state_size}, action_size={self.action_size}")
                    print(f"  Recreating model with saved dimensions...")
                    
                    # Update dimensions and recreate networks
                    self.state_size = saved_state_size
                    self.action_size = saved_action_size
                    self._create_networks()
            
            self.dqn_model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            if 'step_count' in checkpoint:
                self.step_count = checkpoint['step_count']
            print(f"Model loaded from {filepath}")
        else:
            print(f"Error: Model file {filepath} not found")
    
    def model_save(self, filepath):
        if filepath.endswith('.keras'):
            filepath = filepath[:-6] + '.pt'
        self.save(filepath)
    
    def get_training_stats(self):
        """Get training statistics"""
        return {
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0,
            'memory_size': len(self.memory)
        }