import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from collections import deque
import threading
import time
import os
from .a2c_agent import ActorNetwork, CriticNetwork, AttentionLayer, HierarchicalEncoder, MultiObjectiveRewardNetwork

class SharedAdam(torch.optim.Adam):
    """Shared Adam optimizer for A3C"""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()

class GlobalNetwork(nn.Module):
    """Global shared network for A3C"""
    def __init__(self, state_dim, max_tasks):
        super(GlobalNetwork, self).__init__()
        self.actor = ActorNetwork(state_dim, max_tasks)
        self.critic = CriticNetwork(state_dim)
        
        # Share memory for multiprocessing
        self.actor.share_memory()
        self.critic.share_memory()
    
    def forward(self, state, task_mask=None):
        action_logits, objective_weights, attention_weights = self.actor(state, task_mask)
        value = self.critic(state)
        return action_logits, value, objective_weights, attention_weights

class WorkerAgent:
    """Individual worker agent for A3C"""
    
    def __init__(self, worker_id, global_network, state_dim, max_tasks, env_factory, 
                 device=None, lr=3e-4, gamma=0.99, tau=1.0):
        self.worker_id = worker_id
        self.state_dim = state_dim
        self.max_tasks = max_tasks
        self.device = device if device else torch.device("cpu")  # Workers typically use CPU
        self.gamma = gamma
        self.tau = tau  # Entropy coefficient
        
        # Local networks (copies of global)
        self.local_network = GlobalNetwork(state_dim, max_tasks).to(self.device)
        self.global_network = global_network
        
        # Environment factory (function that creates environment instance)
        self.env_factory = env_factory
        
        # Optimizer for global network
        self.optimizer = SharedAdam(
            list(global_network.actor.parameters()) + list(global_network.critic.parameters()),
            lr=lr
        )
        
        # Experience buffer
        self.reset_episode()
        
        # Metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.total_updates = 0
        
        print(f"Worker {worker_id} initialized on device: {self.device}")
    
    def reset_episode(self):
        """Reset episode data"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def sync_with_global(self):
        """Synchronize local network with global network"""
        self.local_network.load_state_dict(self.global_network.state_dict())
    
    def act(self, state, task_mask=None, deterministic=False):
        """Select action using local network"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        if task_mask is not None and not isinstance(task_mask, torch.Tensor):
            task_mask = torch.FloatTensor(task_mask).to(self.device)
            if len(task_mask.shape) == 1:
                task_mask = task_mask.unsqueeze(0)
        
        with torch.no_grad():
            action_logits, value, objective_weights, attention_weights = self.local_network(state, task_mask)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
                log_prob = F.log_softmax(action_logits, dim=-1)[0, action]
            else:
                action_probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item(), {
            'objective_weights': objective_weights.cpu().numpy(),
            'attention_weights': attention_weights.cpu().numpy()
        }
    
    def remember(self, state, action, reward, log_prob, value, done):
        """Store experience"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns(self, next_value=0):
        """Compute discounted returns"""
        returns = []
        R = next_value
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        
        return returns
    
    def update_global(self, next_state):
        """Update global network using accumulated experience"""
        if len(self.states) == 0:
            return {}
        
        # Sync with global network
        self.sync_with_global()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        
        # Compute bootstrap value for next state
        if next_state is not None and not self.dones[-1]:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value, _, _ = self.local_network(next_state_tensor)
                next_value = next_value.item()
        else:
            next_value = 0
        
        # Compute returns and advantages
        returns = self.compute_returns(next_value)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - values
        
        # Forward pass through local network
        action_logits, pred_values, objective_weights, attention_weights = self.local_network(states)
        
        # Policy loss
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(pred_values.squeeze(), returns)
        
        # Entropy loss (for exploration)
        entropy = -(F.softmax(action_logits, dim=-1) * F.log_softmax(action_logits, dim=-1)).sum(dim=-1).mean()
        entropy_loss = -self.tau * entropy
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss
        
        # Update global network
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            list(self.global_network.actor.parameters()) + list(self.global_network.critic.parameters()),
            0.5
        )
        
        self.optimizer.step()
        self.total_updates += 1
        
        # Clear episode data
        self.reset_episode()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item()
        }
    
    def run_episode(self, max_steps=1000, update_frequency=20):
        """Run a single episode"""
        env = self.env_factory()
        state, _ = env.reset()
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Get valid actions
            valid_actions = env.get_valid_actions() if hasattr(env, 'get_valid_actions') else None
            
            # Create task mask
            task_mask = None
            if valid_actions is not None:
                task_mask = torch.zeros(self.max_tasks)
                task_mask[valid_actions] = 1.0
            
            # Select action
            action, log_prob, value, info = self.act(state, task_mask)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience
            self.remember(state, action, reward, log_prob, value, done or truncated)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Update global network periodically or at episode end
            if len(self.states) >= update_frequency or done or truncated:
                self.update_global(next_state if not (done or truncated) else None)
            
            if done or truncated:
                break
        
        # Record episode metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return episode_reward, episode_length
    
    def train(self, num_episodes, max_steps_per_episode=1000, update_frequency=20):
        """Training loop for worker"""
        print(f"Worker {self.worker_id} starting training...")
        
        for episode in range(num_episodes):
            episode_reward, episode_length = self.run_episode(max_steps_per_episode, update_frequency)
            
            # Log progress periodically
            if episode % 10 == 0:
                avg_reward = np.mean(list(self.episode_rewards)[-10:])
                avg_length = np.mean(list(self.episode_lengths)[-10:])
                print(f"Worker {self.worker_id} - Episode {episode}: "
                      f"Reward={episode_reward:.2f}, Length={episode_length}, "
                      f"Avg Reward={avg_reward:.2f}, Avg Length={avg_length:.1f}")
        
        print(f"Worker {self.worker_id} completed training")

class A3CAgent:
    """Asynchronous Advantage Actor-Critic Agent Master"""
    
    def __init__(self, state_dim, max_tasks, num_workers=4, device=None):
        self.state_dim = state_dim
        self.max_tasks = max_tasks
        self.num_workers = num_workers
        self.device = device if device else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Global network
        self.global_network = GlobalNetwork(state_dim, max_tasks).to(self.device)
        
        # Worker processes will be created during training
        self.workers = []
        self.processes = []
        
        # Training metrics (collected from workers)
        self.global_episode_rewards = deque(maxlen=1000)
        self.global_episode_lengths = deque(maxlen=1000)
        
        print(f"A3C Master initialized with {num_workers} workers on device: {self.device}")
    
    def create_worker(self, worker_id, env_factory):
        """Create a worker agent"""
        return WorkerAgent(
            worker_id=worker_id,
            global_network=self.global_network,
            state_dim=self.state_dim,
            max_tasks=self.max_tasks,
            env_factory=env_factory,
            device=torch.device("cpu"),  # Workers use CPU
            lr=3e-4
        )
    
    def worker_process(self, worker_id, env_factory, num_episodes, max_steps_per_episode, update_frequency, results_queue):
        """Worker process function"""
        try:
            worker = self.create_worker(worker_id, env_factory)
            worker.train(num_episodes, max_steps_per_episode, update_frequency)
            
            # Send results back to main process
            results_queue.put({
                'worker_id': worker_id,
                'episode_rewards': list(worker.episode_rewards),
                'episode_lengths': list(worker.episode_lengths),
                'total_updates': worker.total_updates
            })
            
        except Exception as e:
            print(f"Worker {worker_id} encountered error: {e}")
            results_queue.put({'worker_id': worker_id, 'error': str(e)})
    
    def train(self, env_factory, num_episodes_per_worker=1000, max_steps_per_episode=1000, update_frequency=20):
        """Train A3C with multiple workers"""
        print(f"Starting A3C training with {self.num_workers} workers...")
        
        # Create result queue for collecting worker results
        results_queue = mp.Queue()
        
        # Start worker processes
        processes = []
        for worker_id in range(self.num_workers):
            p = mp.Process(
                target=self.worker_process,
                args=(worker_id, env_factory, num_episodes_per_worker, 
                      max_steps_per_episode, update_frequency, results_queue)
            )
            p.start()
            processes.append(p)
            print(f"Started worker {worker_id}")
        
        # Monitor progress
        completed_workers = 0
        while completed_workers < self.num_workers:
            try:
                result = results_queue.get(timeout=10)
                
                if 'error' in result:
                    print(f"Worker {result['worker_id']} failed: {result['error']}")
                else:
                    print(f"Worker {result['worker_id']} completed training: "
                          f"{len(result['episode_rewards'])} episodes, "
                          f"{result['total_updates']} updates")
                    
                    # Collect metrics
                    self.global_episode_rewards.extend(result['episode_rewards'])
                    self.global_episode_lengths.extend(result['episode_lengths'])
                
                completed_workers += 1
                
            except:
                # Check if processes are still alive
                alive_count = sum(1 for p in processes if p.is_alive())
                print(f"Waiting for workers... {alive_count} still running")
        
        # Clean up processes
        for p in processes:
            p.join()
        
        print("A3C training completed!")
        
        # Training summary
        if self.global_episode_rewards:
            print(f"Final Results:")
            print(f"  Total Episodes: {len(self.global_episode_rewards)}")
            print(f"  Average Reward: {np.mean(self.global_episode_rewards):.3f}")
            print(f"  Average Length: {np.mean(self.global_episode_lengths):.1f}")
            print(f"  Final 100 Episodes Avg Reward: {np.mean(list(self.global_episode_rewards)[-100:]):.3f}")
    
    def act(self, state, task_mask=None, deterministic=True):
        """Use global network for action selection during evaluation"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        if task_mask is not None and not isinstance(task_mask, torch.Tensor):
            task_mask = torch.FloatTensor(task_mask).to(self.device)
            if len(task_mask.shape) == 1:
                task_mask = task_mask.unsqueeze(0)
        
        with torch.no_grad():
            action_logits, value, objective_weights, attention_weights = self.global_network(state, task_mask)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action_probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
        
        return action.item(), {
            'value': value.item(),
            'objective_weights': objective_weights.cpu().numpy(),
            'attention_weights': attention_weights.cpu().numpy()
        }
    
    def save(self, filepath):
        """Save global network"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if not filepath.endswith('.pt'):
            filepath = f"{filepath}.pt"
        
        torch.save({
            'global_network_state_dict': self.global_network.state_dict(),
            'hyperparameters': {
                'state_dim': self.state_dim,
                'max_tasks': self.max_tasks,
                'num_workers': self.num_workers
            },
            'training_stats': {
                'episode_rewards': list(self.global_episode_rewards),
                'episode_lengths': list(self.global_episode_lengths)
            }
        }, filepath)
        
        print(f"A3C model saved to {filepath}")
    
    def load(self, filepath):
        """Load global network"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.global_network.load_state_dict(checkpoint['global_network_state_dict'])
            
            if 'training_stats' in checkpoint:
                self.global_episode_rewards.extend(checkpoint['training_stats']['episode_rewards'])
                self.global_episode_lengths.extend(checkpoint['training_stats']['episode_lengths'])
            
            print(f"A3C model loaded from {filepath}")
        else:
            print(f"Error: Model file {filepath} not found")
    
    def get_training_stats(self):
        """Get training statistics"""
        return {
            'num_workers': self.num_workers,
            'total_episodes': len(self.global_episode_rewards),
            'avg_reward': np.mean(self.global_episode_rewards) if self.global_episode_rewards else 0,
            'avg_length': np.mean(self.global_episode_lengths) if self.global_episode_lengths else 0,
            'recent_avg_reward': np.mean(list(self.global_episode_rewards)[-100:]) if len(self.global_episode_rewards) >= 100 else 0
        }

# Worker training function for multiprocessing
def train_worker(worker_id, global_network, state_dim, max_tasks, env_factory, 
                num_episodes, max_steps_per_episode, update_frequency, results_queue):
    """Standalone function for worker training (needed for multiprocessing)"""
    try:
        worker = WorkerAgent(
            worker_id=worker_id,
            global_network=global_network,
            state_dim=state_dim,
            max_tasks=max_tasks,
            env_factory=env_factory,
            device=torch.device("cpu"),
            lr=3e-4
        )
        
        worker.train(num_episodes, max_steps_per_episode, update_frequency)
        
        results_queue.put({
            'worker_id': worker_id,
            'episode_rewards': list(worker.episode_rewards),
            'episode_lengths': list(worker.episode_lengths),
            'total_updates': worker.total_updates
        })
        
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        results_queue.put({'worker_id': worker_id, 'error': str(e)})