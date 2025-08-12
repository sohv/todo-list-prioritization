import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os

class AttentionLayer(nn.Module):
    """Attention mechanism to focus on relevant tasks"""
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Linear(input_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, task_features, mask=None):
        """
        Args:
            task_features: [batch_size, num_tasks, feature_dim]
            mask: [batch_size, num_tasks] - 1 for valid tasks, 0 for padding
        """
        # Compute attention scores
        attention_scores = self.v(torch.tanh(self.W(task_features)))  # [batch, num_tasks, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch, num_tasks]
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -float('inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, num_tasks]
        
        # Apply attention to task features
        attended_features = torch.sum(
            task_features * attention_weights.unsqueeze(-1), 
            dim=1
        )  # [batch, feature_dim]
        
        return attended_features, attention_weights

class HierarchicalEncoder(nn.Module):
    """Hierarchical encoding of tasks (project/category -> individual tasks)"""
    def __init__(self, task_feature_dim, category_dim, hidden_dim):
        super(HierarchicalEncoder, self).__init__()
        
        # Task-level encoding
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Category-level encoding
        self.category_encoder = nn.Sequential(
            nn.Linear(category_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Hierarchical fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, task_features, category_features):
        """
        Args:
            task_features: [batch_size, num_tasks, task_feature_dim]
            category_features: [batch_size, num_tasks, category_dim]
        """
        task_encoded = self.task_encoder(task_features)
        category_encoded = self.category_encoder(category_features)
        
        # Concatenate and fuse
        combined = torch.cat([task_encoded, category_encoded], dim=-1)
        hierarchical_features = self.fusion_layer(combined)
        
        return hierarchical_features

class MultiObjectiveRewardNetwork(nn.Module):
    """Network to learn multi-objective reward weighting"""
    def __init__(self, state_dim, num_objectives=4):
        super(MultiObjectiveRewardNetwork, self).__init__()
        self.num_objectives = num_objectives
        
        self.weight_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_objectives),
            nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )
        
    def forward(self, state):
        """Learn context-dependent objective weights"""
        return self.weight_network(state)

class ActorNetwork(nn.Module):
    """Actor network for continuous task prioritization"""
    def __init__(self, state_dim, max_tasks, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.max_tasks = max_tasks
        
        # Task feature extraction
        task_feature_dim = 4  # priority, urgency, efficiency, estimated_time
        category_dim = 16     # Learned category embeddings
        
        # Hierarchical encoding
        self.hierarchical_encoder = HierarchicalEncoder(
            task_feature_dim, category_dim, hidden_dim // 2
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim // 2, hidden_dim // 4)
        
        # Context encoding
        self.context_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-objective reward weighting
        self.reward_weighter = MultiObjectiveRewardNetwork(hidden_dim)
        
        # Final policy layers
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_tasks)
        )
        
        # Category embedding layer
        self.category_embeddings = nn.Embedding(10, category_dim)  # Assume 10 categories
        
    def forward(self, state, task_mask=None):
        """
        Args:
            state: [batch_size, state_dim]
            task_mask: [batch_size, max_tasks] - 1 for valid tasks
        """
        batch_size = state.shape[0]
        
        # For simplicity, use the full state as context and create dummy task features
        # In a real implementation, state would be parsed to extract task-specific features
        context_features = self.context_encoder(state)
        
        # Create simple task features from state (simplified approach)
        # Extract the action mask part of the state if available
        if state.shape[1] >= self.max_tasks * 5:  # Assuming state includes mask
            mask_start = self.max_tasks * 4
            task_mask_from_state = state[:, mask_start:mask_start + self.max_tasks]
            
            # Create basic task features from available information
            task_features = state[:, :self.max_tasks * 4].view(batch_size, self.max_tasks, 4)
        else:
            # Fallback: create dummy features
            task_features = torch.randn(batch_size, self.max_tasks, 4, device=state.device)
        
        # Generate simple category features
        category_ids = torch.randint(0, 10, (batch_size, self.max_tasks), device=state.device)
        category_features = self.category_embeddings(category_ids)
        
        # Hierarchical encoding
        hierarchical_features = self.hierarchical_encoder(task_features, category_features)
        
        # Apply attention
        attended_features, attention_weights = self.attention(hierarchical_features, task_mask)
        
        # Multi-objective weighting
        objective_weights = self.reward_weighter(context_features)
        
        # Combine features
        combined_features = torch.cat([context_features, attended_features], dim=-1)
        
        # Generate action logits
        action_logits = self.policy_head(combined_features)
        
        # Apply mask to invalid actions
        if task_mask is not None:
            action_logits = action_logits.masked_fill(task_mask == 0, -float('inf'))
        
        return action_logits, objective_weights, attention_weights

class CriticNetwork(nn.Module):
    """Critic network for value estimation"""
    def __init__(self, state_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.value_head(state)

class A2CAgent:
    """Advantage Actor-Critic Agent for Task Prioritization"""
    
    def __init__(self, state_dim, max_tasks, device=None, lr_actor=3e-4, lr_critic=1e-3):
        self.state_dim = state_dim
        self.max_tasks = max_tasks
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = ActorNetwork(state_dim, max_tasks).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Experience storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
        # Training parameters
        self.gamma = 0.99
        self.entropy_coeff = 0.01
        self.value_loss_coeff = 0.5
        self.max_grad_norm = 0.5
        
        # Metrics
        self.actor_losses = deque(maxlen=1000)
        self.critic_losses = deque(maxlen=1000)
        self.entropy_losses = deque(maxlen=1000)
        
        print(f"A2C Agent initialized on device: {self.device}")
    
    def act(self, state, task_mask=None, deterministic=False):
        """Select action using the actor network"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        if task_mask is not None and not isinstance(task_mask, torch.Tensor):
            task_mask = torch.FloatTensor(task_mask).to(self.device)
            if len(task_mask.shape) == 1:
                task_mask = task_mask.unsqueeze(0)
        
        with torch.no_grad():
            action_logits, objective_weights, attention_weights = self.actor(state, task_mask)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
                log_prob = F.log_softmax(action_logits, dim=-1)[0, action]
            else:
                action_probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), {
            'objective_weights': objective_weights.cpu().numpy(),
            'attention_weights': attention_weights.cpu().numpy()
        }
    
    def remember(self, state, action, reward, next_state, done, log_prob):
        """Store experience"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
    
    def compute_returns(self, next_value=0):
        """Compute discounted returns"""
        returns = []
        R = next_value
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        
        return returns
    
    def update(self):
        """Update actor and critic networks"""
        if len(self.states) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Compute returns and advantages
        with torch.no_grad():
            if not self.dones[-1]:  # If last state is not terminal
                next_value = self.critic(torch.FloatTensor(self.next_states[-1]).unsqueeze(0).to(self.device)).item()
            else:
                next_value = 0
        
        returns = self.compute_returns(next_value)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Get current values
        values = self.critic(states).squeeze()
        advantages = returns - values
        
        # Actor loss (policy gradient with entropy regularization)
        action_logits, objective_weights, attention_weights = self.actor(states)
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Policy loss
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        
        # Entropy loss (for exploration)
        entropy = -(F.softmax(action_logits, dim=-1) * F.log_softmax(action_logits, dim=-1)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coeff * entropy
        
        # Total actor loss
        actor_loss = policy_loss + entropy_loss
        
        # Critic loss
        critic_loss = F.mse_loss(values, returns)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.entropy_losses.append(entropy.item())
        
        # Clear experience buffer
        self.clear_memory()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item()
        }
    
    def clear_memory(self):
        """Clear experience buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
    
    def save(self, filepath):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if not filepath.endswith('.pt'):
            filepath = f"{filepath}.pt"
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'hyperparameters': {
                'state_dim': self.state_dim,
                'max_tasks': self.max_tasks,
                'gamma': self.gamma,
                'entropy_coeff': self.entropy_coeff,
                'value_loss_coeff': self.value_loss_coeff
            }
        }, filepath)
        
        print(f"A2C model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            print(f"A2C model loaded from {filepath}")
        else:
            print(f"Error: Model file {filepath} not found")
    
    def get_training_stats(self):
        """Get training statistics"""
        return {
            'actor_loss': np.mean(self.actor_losses) if self.actor_losses else 0,
            'critic_loss': np.mean(self.critic_losses) if self.critic_losses else 0,
            'entropy': np.mean(self.entropy_losses) if self.entropy_losses else 0,
            'memory_size': len(self.states)
        }