import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch

class RLEvaluator:
    """Comprehensive evaluation utilities for RL agents"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.evaluation_results = {}
        
    def evaluate_agent_performance(self, agent, env, num_episodes: int = 100) -> Dict[str, Any]:
        """Comprehensive agent evaluation"""
        print(f"📊 Evaluating agent for {num_episodes} episodes...")
        
        rewards = []
        episode_lengths = []
        final_states = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1000:
                action = agent.choose_action(state, training=False)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                steps += 1
            
            rewards.append(episode_reward)
            episode_lengths.append(steps)
            final_states.append(state)
        
        # Calculate statistics
        results = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'rewards': rewards,
            'episode_lengths': episode_lengths
        }
        
        # Calculate additional metrics
        results['stability'] = self._calculate_stability(rewards)
        results['learning_efficiency'] = self._calculate_learning_efficiency(rewards)
        results['convergence_episode'] = self._find_convergence_episode(rewards)
        
        return results
    
    def _calculate_stability(self, rewards: List[float], window_size: int = 50) -> float:
        """Calculate reward stability (inverse of variance)"""
        if len(rewards) < window_size:
            return 0.0
        
        # Calculate rolling standard deviation
        rolling_std = []
        for i in range(window_size, len(rewards)):
            window = rewards[i-window_size:i]
            rolling_std.append(np.std(window))
        
        # Stability is inverse of average rolling std
        avg_std = np.mean(rolling_std)
        return 1.0 / (1.0 + avg_std)
    
    def _calculate_learning_efficiency(self, rewards: List[float]) -> float:
        """Calculate how quickly the agent learns"""
        if len(rewards) < 10:
            return 0.0
        
        # Calculate improvement rate
        first_quarter = rewards[:len(rewards)//4]
        last_quarter = rewards[-len(rewards)//4:]
        
        improvement = np.mean(last_quarter) - np.mean(first_quarter)
        return max(0.0, improvement)
    
    def _find_convergence_episode(self, rewards: List[float], window_size: int = 50, threshold: float = 0.1) -> int:
        """Find episode where rewards converge"""
        if len(rewards) < window_size * 2:
            return len(rewards)
        
        for i in range(window_size, len(rewards) - window_size):
            window1 = rewards[i-window_size:i]
            window2 = rewards[i:i+window_size]
            
            if abs(np.mean(window1) - np.mean(window2)) < threshold:
                return i
        
        return len(rewards)
    
    def compare_agents(self, agents: Dict[str, Any], envs: Dict[str, Any], 
                      num_episodes: int = 100) -> Dict[str, Any]:
        """Compare multiple agents across different environments"""
        print("🔄 Comparing agents...")
        
        comparison_results = {}
        
        for agent_name, agent in agents.items():
            if agent_name in envs:
                env = envs[agent_name]
                results = self.evaluate_agent_performance(agent, env, num_episodes)
                comparison_results[agent_name] = results
        
        # Calculate pairwise comparisons
        pairwise_comparisons = {}
        agent_names = list(comparison_results.keys())
        
        for i, agent1 in enumerate(agent_names):
            for j, agent2 in enumerate(agent_names[i+1:], i+1):
                key = f"{agent1}_vs_{agent2}"
                
                # Statistical significance test
                rewards1 = comparison_results[agent1]['rewards']
                rewards2 = comparison_results[agent2]['rewards']
                
                # Simple t-test approximation
                mean_diff = np.mean(rewards1) - np.mean(rewards2)
                pooled_std = np.sqrt((np.std(rewards1)**2 + np.std(rewards2)**2) / 2)
                t_stat = mean_diff / (pooled_std / np.sqrt(len(rewards1)))
                
                pairwise_comparisons[key] = {
                    'mean_difference': mean_diff,
                    't_statistic': t_stat,
                    'better_agent': agent1 if mean_diff > 0 else agent2
                }
        
        comparison_results['pairwise_comparisons'] = pairwise_comparisons
        
        return comparison_results
    
    def analyze_training_curves(self, training_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze training curves for multiple agents"""
        print("📈 Analyzing training curves...")
        
        analysis = {}
        
        for agent_name, rewards in training_data.items():
            if len(rewards) == 0:
                continue
            
            # Basic statistics
            analysis[agent_name] = {
                'final_performance': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
                'peak_performance': np.max(rewards),
                'improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0,
                'volatility': np.std(rewards),
                'positive_episodes': sum(1 for r in rewards if r > 0),
                'total_episodes': len(rewards)
            }
            
            # Learning phases
            analysis[agent_name]['learning_phases'] = self._identify_learning_phases(rewards)
        
        return analysis
    
    def _identify_learning_phases(self, rewards: List[float], window_size: int = 50) -> Dict[str, int]:
        """Identify different phases of learning"""
        if len(rewards) < window_size * 2:
            return {'exploration': 0, 'learning': len(rewards), 'exploitation': 0}
        
        # Calculate moving average
        moving_avg = []
        for i in range(window_size, len(rewards)):
            window = rewards[i-window_size:i]
            moving_avg.append(np.mean(window))
        
        # Find phases based on slope
        phases = {'exploration': 0, 'learning': 0, 'exploitation': 0}
        
        # Exploration phase: high volatility, low performance
        exploration_end = window_size
        for i in range(len(moving_avg)):
            if i > window_size and moving_avg[i] > moving_avg[0] * 1.5:
                exploration_end = i + window_size
                break
        
        phases['exploration'] = exploration_end
        
        # Learning phase: improving performance
        learning_end = len(rewards)
        for i in range(exploration_end, len(moving_avg)):
            if i > exploration_end + window_size:
                recent_slope = np.polyfit(range(window_size), moving_avg[i-window_size:i], 1)[0]
                if abs(recent_slope) < 0.01:  # Flat slope
                    learning_end = i + window_size
                    break
        
        phases['learning'] = learning_end - exploration_end
        phases['exploitation'] = len(rewards) - learning_end
        
        return phases
    
    def generate_evaluation_report(self, results: Dict[str, Any], save_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        print("📝 Generating evaluation report...")
        
        report = "# RL Agent Evaluation Report\n\n"
        
        # Overall summary
        report += "## Overall Summary\n\n"
        if 'pairwise_comparisons' in results:
            report += "### Agent Rankings\n\n"
            
            # Rank agents by mean reward
            agent_scores = {}
            for agent_name in results.keys():
                if agent_name != 'pairwise_comparisons':
                    agent_scores[agent_name] = results[agent_name]['mean_reward']
            
            ranked_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
            
            for i, (agent, score) in enumerate(ranked_agents, 1):
                report += f"{i}. **{agent}**: {score:.3f} mean reward\n"
        
        # Detailed results for each agent
        report += "\n## Detailed Results\n\n"
        for agent_name, agent_results in results.items():
            if agent_name == 'pairwise_comparisons':
                continue
            
            report += f"### {agent_name}\n\n"
            report += f"- **Mean Reward**: {agent_results['mean_reward']:.3f} ± {agent_results['std_reward']:.3f}\n"
            
            # Add optional metrics if available
            if 'min_reward' in agent_results and 'max_reward' in agent_results:
                report += f"- **Min/Max Reward**: {agent_results['min_reward']:.3f} / {agent_results['max_reward']:.3f}\n"
            
            if 'mean_episode_length' in agent_results:
                report += f"- **Mean Episode Length**: {agent_results['mean_episode_length']:.1f}\n"
            
            if 'stability' in agent_results:
                report += f"- **Stability**: {agent_results['stability']:.3f}\n"
            
            if 'learning_efficiency' in agent_results:
                report += f"- **Learning Efficiency**: {agent_results['learning_efficiency']:.3f}\n"
            
            if 'convergence_episode' in agent_results:
                report += f"- **Convergence Episode**: {agent_results['convergence_episode']}\n"
            
            report += "\n"
        
        # Pairwise comparisons
        if 'pairwise_comparisons' in results:
            report += "## Pairwise Comparisons\n\n"
            for comparison, stats in results['pairwise_comparisons'].items():
                report += f"### {comparison.replace('_', ' ').title()}\n\n"
                report += f"- **Mean Difference**: {stats['mean_difference']:.3f}\n"
                report += f"- **T-Statistic**: {stats['t_statistic']:.3f}\n"
                report += f"- **Better Agent**: {stats['better_agent']}\n\n"
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report

class Visualizer:
    """Advanced visualization utilities for RL experiments"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_comparison(self, training_data: Dict[str, List[float]], 
                                 save_path: str = None, smooth_window: int = 50):
        """Plot training curves for multiple agents"""
        plt.figure(figsize=(12, 8))
        
        for agent_name, rewards in training_data.items():
            if len(rewards) == 0:
                continue
            
            # Smooth the curve
            if len(rewards) > smooth_window:
                smoothed = np.convolve(rewards, np.ones(smooth_window)/smooth_window, mode='valid')
                x = range(smooth_window-1, len(rewards))
                plt.plot(x, smoothed, label=f'{agent_name} (smoothed)', linewidth=2)
            
            # Plot original data with lower alpha
            plt.plot(rewards, label=f'{agent_name} (raw)', alpha=0.3, linewidth=0.5)
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_distribution(self, evaluation_results: Dict[str, Any], 
                                    save_path: str = None):
        """Plot performance distribution for agents"""
        plt.figure(figsize=(15, 5))
        
        agents = [k for k in evaluation_results.keys() if k != 'pairwise_comparisons']
        
        # Subplot 1: Reward distributions
        plt.subplot(1, 3, 1)
        for agent in agents:
            rewards = evaluation_results[agent]['rewards']
            plt.hist(rewards, alpha=0.7, label=agent, bins=30)
        
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.legend()
        
        # Subplot 2: Box plot comparison
        plt.subplot(1, 3, 2)
        data_for_box = []
        labels_for_box = []
        
        for agent in agents:
            rewards = evaluation_results[agent]['rewards']
            data_for_box.append(rewards)
            labels_for_box.append(agent)
        
        plt.boxplot(data_for_box, labels=labels_for_box)
        plt.ylabel('Reward')
        plt.title('Performance Comparison')
        plt.xticks(rotation=45)
        
        # Subplot 3: Performance metrics
        plt.subplot(1, 3, 3)
        metrics = ['mean_reward', 'stability', 'learning_efficiency']
        metric_data = {metric: [] for metric in metrics}
        
        for agent in agents:
            for metric in metrics:
                metric_data[metric].append(evaluation_results[agent][metric])
        
        x = np.arange(len(agents))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, metric_data[metric], width, label=metric.replace('_', ' ').title())
        
        plt.xlabel('Agent')
        plt.ylabel('Score')
        plt.title('Performance Metrics')
        plt.xticks(x + width, agents, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_phases(self, training_data: Dict[str, List[float]], 
                           save_path: str = None):
        """Visualize learning phases for agents"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (agent_name, rewards) in enumerate(training_data.items()):
            if i >= 4 or len(rewards) == 0:
                continue
            
            ax = axes[i]
            
            # Plot rewards
            ax.plot(rewards, alpha=0.7, label='Rewards')
            
            # Identify and mark learning phases
            window_size = 50
            if len(rewards) > window_size * 2:
                moving_avg = []
                for j in range(window_size, len(rewards)):
                    window = rewards[j-window_size:j]
                    moving_avg.append(np.mean(window))
                
                # Simple phase detection
                phases = {'exploration': window_size, 'learning': len(rewards)//2, 'exploitation': len(rewards)}
                
                # Mark phases
                ax.axvline(x=phases['exploration'], color='red', linestyle='--', alpha=0.7, label='Exploration End')
                ax.axvline(x=phases['learning'], color='orange', linestyle='--', alpha=0.7, label='Learning End')
                
                # Plot moving average
                x_ma = range(window_size, len(rewards))
                ax.plot(x_ma, moving_avg, color='black', linewidth=2, label='Moving Average')
            
            ax.set_title(f'{agent_name} Learning Phases')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(training_data), 4):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self, training_data: Dict[str, List[float]], 
                                    evaluation_results: Dict[str, Any],
                                    save_path: str = None):
        """Create a comprehensive evaluation dashboard"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid specification
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Training curves (top, spanning 3 columns)
        ax1 = fig.add_subplot(gs[0, :3])
        for agent_name, rewards in training_data.items():
            if len(rewards) > 0:
                # Smooth curve
                if len(rewards) > 50:
                    smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
                    x = range(49, len(rewards))
                    ax1.plot(x, smoothed, label=agent_name, linewidth=2)
        
        ax1.set_title('Training Progress')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance metrics (top right)
        ax2 = fig.add_subplot(gs[0, 3])
        agents = [k for k in evaluation_results.keys() if k != 'pairwise_comparisons']
        metrics = ['mean_reward', 'stability', 'learning_efficiency']
        
        x = np.arange(len(agents))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[agent][metric] for agent in agents]
            ax2.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax2.set_title('Performance Metrics')
        ax2.set_ylabel('Score')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(agents, rotation=45)
        ax2.legend()
        
        # 3. Reward distributions (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        for agent in agents:
            rewards = evaluation_results[agent]['rewards']
            ax3.hist(rewards, alpha=0.7, label=agent, bins=20)
        
        ax3.set_title('Reward Distributions')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. Box plot comparison (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        data_for_box = []
        for agent in agents:
            data_for_box.append(evaluation_results[agent]['rewards'])
        
        ax4.boxplot(data_for_box, labels=agents)
        ax4.set_title('Performance Comparison')
        ax4.set_ylabel('Reward')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Convergence analysis (bottom left)
        ax5 = fig.add_subplot(gs[2, :2])
        convergence_episodes = [evaluation_results[agent]['convergence_episode'] for agent in agents]
        ax5.bar(agents, convergence_episodes)
        ax5.set_title('Convergence Episodes')
        ax5.set_ylabel('Episode')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Summary statistics (bottom right)
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        
        # Create summary text
        summary_text = "Summary Statistics\n\n"
        for agent in agents:
            results = evaluation_results[agent]
            summary_text += f"{agent}:\n"
            summary_text += f"  Mean: {results['mean_reward']:.3f}\n"
            summary_text += f"  Std: {results['std_reward']:.3f}\n"
            summary_text += f"  Stability: {results['stability']:.3f}\n\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('RL Agent Evaluation Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
