import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import random
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt

class QLearningAgent:
    """Tabular Q-Learning Agent for Digital Wellbeing System"""
    
    def __init__(self, state_space_size: int, action_space_size: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.01):
        
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table
        self.q_table = np.zeros((state_space_size, action_space_size))
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.exploration_rates = []
        
    def discretize_state(self, state: np.ndarray, bins: int = 5) -> int:
        """Convert continuous state to discrete state index"""
        # Use fewer bins to prevent overflow
        discretized = np.digitize(state, bins=np.linspace(-1, 1, bins))
        discretized = np.clip(discretized, 0, bins-1)
        
        # Use a more compact encoding to prevent overflow
        state_idx = 0
        multiplier = 1
        for i, val in enumerate(discretized):
            if i >= 10:  # Limit to first 10 dimensions to prevent overflow
                break
            state_idx += val * multiplier
            multiplier *= bins
            
        return min(state_idx, self.state_space_size - 1)
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        state_idx = self.discretize_state(state)
        
        if training and random.random() < self.exploration_rate:
            return random.randrange(self.action_space_size)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def update_q_table(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool):
        """Update Q-table using Q-learning update rule"""
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)
        
        # Q-learning update
        best_next_action = np.argmax(self.q_table[next_state_idx])
        td_target = reward + self.discount_factor * self.q_table[next_state_idx][best_next_action] * (not done)
        td_error = td_target - self.q_table[state_idx][action]
        
        self.q_table[state_idx][action] += self.learning_rate * td_error
    
    def decay_exploration(self):
        """Decay exploration rate"""
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int]:
        """Train agent for one episode"""
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = self.choose_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            self.update_q_table(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        self.decay_exploration()
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.exploration_rates.append(self.exploration_rate)
        
        return total_reward, steps
    
    def evaluate(self, env, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate agent performance"""
        total_rewards = []
        total_steps = []
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1000:
                action = self.choose_action(state, training=False)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                steps += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps)
        }

class DQNetwork(nn.Module):
    """Deep Q-Network for DQN Agent"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 64]):
        super(DQNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """Deep Q-Network Agent for Digital Wellbeing System"""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.001, discount_factor: float = 0.95,
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.01, memory_size: int = 10000,
                 batch_size: int = 32, target_update_freq: int = 1000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Neural networks
        self.q_network = DQNetwork(state_size, action_size)
        self.target_network = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network
        self.update_target_network()
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.exploration_rates = []
        self.losses = []
        self.step_count = 0
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.exploration_rate:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def replay_experience(self) -> float:
        """Train network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def decay_exploration(self):
        """Decay exploration rate"""
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int]:
        """Train agent for one episode"""
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_loss = 0
        
        for step in range(max_steps):
            action = self.choose_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            self.remember(state, action, reward, next_state, done)
            
            # Replay experience and train
            loss = self.replay_experience()
            episode_loss += loss
            
            state = next_state
            total_reward += reward
            steps += 1
            self.step_count += 1
            
            # Update target network
            if self.step_count % self.target_update_freq == 0:
                self.update_target_network()
            
            if done:
                break
        
        self.decay_exploration()
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.exploration_rates.append(self.exploration_rate)
        if steps > 0:
            self.losses.append(episode_loss / steps)
        
        return total_reward, steps
    
    def evaluate(self, env, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate agent performance"""
        total_rewards = []
        total_steps = []
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1000:
                action = self.choose_action(state, training=False)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                steps += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps)
        }
    
    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'step_count': self.step_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.step_count = checkpoint['step_count']

def plot_training_results(agents: Dict[str, Any], save_path: str = None):
    """Plot training results for comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for name, agent in agents.items():
        if hasattr(agent, 'episode_rewards'):
            # Smooth the rewards for better visualization
            rewards = agent.episode_rewards
            smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
            
            axes[0, 0].plot(range(len(smoothed_rewards)), smoothed_rewards, label=name)
            axes[0, 1].plot(agent.exploration_rates, label=name)
            
            if hasattr(agent, 'losses') and agent.losses:
                axes[1, 0].plot(agent.losses, label=name)
    
    axes[0, 0].set_title('Episode Rewards (Smoothed)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    
    axes[0, 1].set_title('Exploration Rate')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].legend()
    
    axes[1, 0].set_title('Training Loss (DQN only)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    
    # Remove empty subplot
    axes[1, 1].remove()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
