import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, Any

# Add current directory to path first to ensure correct environment import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add parent directory to path for utils imports
sys.path.append(os.path.dirname(current_dir))

from utils.data_processor import DataProcessor
from utils.rl_algorithms import QLearningAgent, DQNAgent, plot_training_results
from wellbeing_environment import DigitalWellbeingOptimizationEnvironment

def train_wellbeing_optimization():
    """Train both Q-Learning and DQN agents for Digital Wellbeing Optimization"""
    
    print("🚀 Starting Digital Wellbeing Optimization System Training...")
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    data_processor = DataProcessor(os.path.join(os.path.dirname(current_dir), 'Mobile_Usage_Screentime_Dataset_.xlsx'))
    data_processor.load_data()
    user_profiles = data_processor.create_user_profiles()
    
    print(f"✅ Loaded {len(user_profiles)} user profiles")
    
    # Create environment
    env = DigitalWellbeingOptimizationEnvironment(user_profiles)
    
    # Initialize agents
    state_size = env.state_size
    action_size = env.action_space.n
    
    print(f"🎮 Environment: State size={state_size}, Action size={action_size}")
    
    # Q-Learning Agent
    print("\n📚 Initializing Q-Learning Agent...")
    q_agent = QLearningAgent(
        state_space_size=1000,  # Discretized state space
        action_space_size=action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01
    )
    
    # DQN Agent
    print("🧠 Initializing DQN Agent...")
    dqn_agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        memory_size=10000,
        batch_size=32,
        target_update_freq=1000
    )
    
    # Training parameters
    num_episodes = 500  # Fewer episodes due to longer episodes
    max_steps_per_episode = 24 * 7  # One week
    
    print(f"\n🏋️ Training for {num_episodes} episodes...")
    
    # Train Q-Learning Agent
    print("\n📈 Training Q-Learning Agent...")
    q_rewards = []
    for episode in range(num_episodes):
        reward, steps = q_agent.train_episode(env, max_steps_per_episode)
        q_rewards.append(reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(q_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 50): {avg_reward:.2f}")
    
    # Train DQN Agent
    print("\n🧠 Training DQN Agent...")
    dqn_rewards = []
    for episode in range(num_episodes):
        reward, steps = dqn_agent.train_episode(env, max_steps_per_episode)
        dqn_rewards.append(reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(dqn_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 50): {avg_reward:.2f}")
    
    # Evaluate both agents
    print("\n📊 Evaluating agents...")
    
    q_eval = q_agent.evaluate(env, num_episodes=50)
    dqn_eval = dqn_agent.evaluate(env, num_episodes=50)
    
    print(f"\n🏆 Q-Learning Performance:")
    print(f"   Mean Reward: {q_eval['mean_reward']:.2f} ± {q_eval['std_reward']:.2f}")
    print(f"   Mean Steps: {q_eval['mean_steps']:.2f} ± {q_eval['std_steps']:.2f}")
    
    print(f"\n🏆 DQN Performance:")
    print(f"   Mean Reward: {dqn_eval['mean_reward']:.2f} ± {dqn_eval['std_reward']:.2f}")
    print(f"   Mean Steps: {dqn_eval['mean_steps']:.2f} ± {dqn_eval['std_steps']:.2f}")
    
    # Plot training results
    print("\n📈 Plotting training results...")
    agents = {'Q-Learning': q_agent, 'DQN': dqn_agent}
    plot_training_results(agents, os.path.join(os.path.dirname(current_dir), 'results/wellbeing_optimization_training.png'))
    
    # Save models
    print("\n💾 Saving models...")
    dqn_agent.save_model(os.path.join(os.path.dirname(current_dir), 'results/models/wellbeing_optimization_dqn_model.pth'))
    
    # Save training data
    np.save(os.path.join(os.path.dirname(current_dir), 'results/models/wellbeing_optimization_q_rewards.npy'), q_rewards)
    np.save(os.path.join(os.path.dirname(current_dir), 'results/models/wellbeing_optimization_dqn_rewards.npy'), dqn_rewards)
    
    # Test with specific scenarios
    print("\n🧪 Testing specific scenarios...")
    test_wellbeing_scenarios(env, q_agent, dqn_agent, user_profiles)
    
    # Analyze intervention effectiveness
    print("\n📊 Analyzing intervention effectiveness...")
    analyze_intervention_effectiveness(env, dqn_agent, user_profiles)
    
    print("\n✅ Digital Wellbeing Optimization System Training Complete!")
    
    return {
        'q_agent': q_agent,
        'dqn_agent': dqn_agent,
        'environment': env,
        'q_eval': q_eval,
        'dqn_eval': dqn_eval
    }

def test_wellbeing_scenarios(env, q_agent, dqn_agent, user_profiles):
    """Test agents with specific wellbeing scenarios"""
    
    print("\n🎯 Testing specific wellbeing scenarios...")
    
    # Test different user types
    user_types = {}
    for user_id, profile in user_profiles.items():
        usage_pattern = profile['usage_pattern']
        if usage_pattern not in user_types:
            user_types[usage_pattern] = user_id
    
    for usage_type, user_id in user_types.items():
        print(f"\n📱 Testing {usage_type} user (ID: {user_id})...")
        
        # Test Q-Learning
        state = env.reset()
        env.current_user_id = user_id  # Force specific user
        
        q_total_reward = 0
        q_actions = []
        
        for step in range(24 * 3):  # 3 days
            action = q_agent.choose_action(state, training=False)
            q_actions.append(action)
            state, reward, done, info = env.step(action)
            q_total_reward += reward
            
            if done:
                break
        
        # Test DQN
        state = env.reset()
        env.current_user_id = user_id  # Force specific user
        
        dqn_total_reward = 0
        dqn_actions = []
        
        for step in range(24 * 3):  # 3 days
            action = dqn_agent.choose_action(state, training=False)
            dqn_actions.append(action)
            state, reward, done, info = env.step(action)
            dqn_total_reward += reward
            
            if done:
                break
        
        print(f"   Q-Learning: Total Reward = {q_total_reward:.2f}")
        print(f"   DQN: Total Reward = {dqn_total_reward:.2f}")
        print(f"   Final Wellbeing Summary: {env.get_wellbeing_summary()}")

def analyze_intervention_effectiveness(env, agent, user_profiles, num_episodes=30):
    """Analyze which interventions work best for different user types"""
    
    print("\n📊 Analyzing intervention effectiveness across user types...")
    
    intervention_stats = {}
    
    for user_id, profile in user_profiles.items():
        usage_pattern = profile['usage_pattern']
        
        if usage_pattern not in intervention_stats:
            intervention_stats[usage_pattern] = {
                'total_episodes': 0,
                'intervention_counts': {},
                'intervention_rewards': {},
                'final_wellbeing': []
            }
        
        # Run multiple episodes for this user
        for episode in range(5):  # 5 episodes per user
            state = env.reset()
            env.current_user_id = user_id
            
            episode_interventions = []
            episode_rewards = []
            
            for step in range(24 * 3):  # 3 days
                action = agent.choose_action(state, training=False)
                state, reward, done, info = env.step(action)
                
                if info['intervention'] != 'none':
                    episode_interventions.append(info['intervention'])
                    episode_rewards.append(reward)
                
                if done:
                    break
            
            # Update statistics
            intervention_stats[usage_pattern]['total_episodes'] += 1
            
            for intervention in set(episode_interventions):
                if intervention not in intervention_stats[usage_pattern]['intervention_counts']:
                    intervention_stats[usage_pattern]['intervention_counts'][intervention] = 0
                    intervention_stats[usage_pattern]['intervention_rewards'][intervention] = []
                
                intervention_stats[usage_pattern]['intervention_counts'][intervention] += 1
                
                # Calculate average reward when this intervention was used
                intervention_rewards = [r for i, r in zip(episode_interventions, episode_rewards) 
                                      if i == intervention]
                if intervention_rewards:
                    intervention_stats[usage_pattern]['intervention_rewards'][intervention].extend(
                        intervention_rewards
                    )
            
            # Store final wellbeing
            final_summary = env.get_wellbeing_summary()
            if final_summary:
                intervention_stats[usage_pattern]['final_wellbeing'].append(
                    final_summary['overall_score']
                )
    
    # Print analysis results
    for usage_pattern, stats in intervention_stats.items():
        print(f"\n🎯 {usage_pattern.title()} Users:")
        print(f"   Episodes: {stats['total_episodes']}")
        print(f"   Average Final Wellbeing: {np.mean(stats['final_wellbeing']):.3f}")
        
        print("   Top Interventions:")
        
        # Calculate intervention effectiveness
        intervention_effectiveness = {}
        for intervention in stats['intervention_counts']:
            count = stats['intervention_counts'][intervention]
            rewards = stats['intervention_rewards'][intervention]
            if rewards:
                avg_reward = np.mean(rewards)
                effectiveness = avg_reward * count  # Weight by usage
                intervention_effectiveness[intervention] = effectiveness
        
        # Sort by effectiveness
        sorted_interventions = sorted(intervention_effectiveness.items(), 
                                    key=lambda x: x[1], reverse=True)
        
        for intervention, effectiveness in sorted_interventions[:5]:
            count = stats['intervention_counts'][intervention]
            rewards = stats['intervention_rewards'][intervention]
            avg_reward = np.mean(rewards) if rewards else 0
            print(f"     {intervention}: {count} times, avg reward: {avg_reward:.3f}")

def create_wellbeing_dashboard(env, agent, num_episodes=10):
    """Create a comprehensive wellbeing dashboard"""
    
    print("\n📊 Creating wellbeing dashboard...")
    
    dashboard_data = {
        'overall_wellbeing': [],
        'dimension_scores': {dim: [] for dim in env.wellbeing_dimensions},
        'intervention_usage': {interv: 0 for interv in env.intervention_types},
        'usage_metrics': {metric: [] for metric in env.usage_metrics.keys()},
        'goal_achievement': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state, training=False)
            state, reward, done, info = env.step(action)
        
        # Collect final episode data
        summary = env.get_wellbeing_summary()
        if summary:
            dashboard_data['overall_wellbeing'].append(summary['overall_score'])
            
            for dim in env.wellbeing_dimensions:
                dashboard_data['dimension_scores'][dim].append(
                    summary['dimension_scores'][dim]
                )
            
            for metric in env.usage_metrics.keys():
                dashboard_data['usage_metrics'][metric].append(
                    summary['usage_metrics'][metric]
                )
            
            dashboard_data['goal_achievement'].append(
                summary['goals_achieved'] / summary['total_goals']
            )
    
    # Print dashboard summary
    print("\n📈 Wellbeing Dashboard Summary:")
    print(f"   Average Overall Wellbeing: {np.mean(dashboard_data['overall_wellbeing']):.3f}")
    
    print("\n   Dimension Scores:")
    for dim in env.wellbeing_dimensions:
        avg_score = np.mean(dashboard_data['dimension_scores'][dim])
        print(f"     {dim}: {avg_score:.3f}")
    
    print(f"\n   Goal Achievement Rate: {np.mean(dashboard_data['goal_achievement']):.3f}")
    
    print("\n   Usage Metrics:")
    for metric, values in dashboard_data['usage_metrics'].items():
        avg_value = np.mean(values)
        print(f"     {metric}: {avg_value:.2f}")
    
    return dashboard_data

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    
    # Run training
    results = train_wellbeing_optimization()
    
    # Create dashboard
    dashboard = create_wellbeing_dashboard(results['environment'], results['dqn_agent'])
