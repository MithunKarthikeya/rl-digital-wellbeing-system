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
from notification_environment import NotificationSchedulingEnvironment

def train_notification_scheduling():
    """Train both Q-Learning and DQN agents for Notification Scheduling"""
    
    print("🚀 Starting Intelligent Notification Scheduling System Training...")
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    data_processor = DataProcessor(os.path.join(os.path.dirname(current_dir), 'Mobile_Usage_Screentime_Dataset_.xlsx'))
    data_processor.load_data()
    user_profiles = data_processor.create_user_profiles()
    
    print(f"✅ Loaded {len(user_profiles)} user profiles")
    
    # Create environment
    env = NotificationSchedulingEnvironment(user_profiles)
    
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
    num_episodes = 1000
    max_steps_per_episode = 24 * 7  # One week
    
    print(f"\n🏋️ Training for {num_episodes} episodes...")
    
    # Train Q-Learning Agent
    print("\n📈 Training Q-Learning Agent...")
    q_rewards = []
    for episode in range(num_episodes):
        reward, steps = q_agent.train_episode(env, max_steps_per_episode)
        q_rewards.append(reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(q_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 100): {avg_reward:.2f}")
    
    # Train DQN Agent
    print("\n🧠 Training DQN Agent...")
    dqn_rewards = []
    for episode in range(num_episodes):
        reward, steps = dqn_agent.train_episode(env, max_steps_per_episode)
        dqn_rewards.append(reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(dqn_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 100): {avg_reward:.2f}")
    
    # Evaluate both agents
    print("\n📊 Evaluating agents...")
    
    q_eval = q_agent.evaluate(env, num_episodes=100)
    dqn_eval = dqn_agent.evaluate(env, num_episodes=100)
    
    print(f"\n🏆 Q-Learning Performance:")
    print(f"   Mean Reward: {q_eval['mean_reward']:.2f} ± {q_eval['std_reward']:.2f}")
    print(f"   Mean Steps: {q_eval['mean_steps']:.2f} ± {q_eval['std_steps']:.2f}")
    
    print(f"\n🏆 DQN Performance:")
    print(f"   Mean Reward: {dqn_eval['mean_reward']:.2f} ± {dqn_eval['std_reward']:.2f}")
    print(f"   Mean Steps: {dqn_eval['mean_steps']:.2f} ± {dqn_eval['std_steps']:.2f}")
    
    # Plot training results
    print("\n📈 Plotting training results...")
    agents = {'Q-Learning': q_agent, 'DQN': dqn_agent}
    plot_training_results(agents, os.path.join(os.path.dirname(current_dir), 'results/notification_scheduling_training.png'))
    
    # Save models
    print("\n💾 Saving models...")
    dqn_agent.save_model(os.path.join(os.path.dirname(current_dir), 'results/models/notification_scheduling_dqn_model.pth'))
    
    # Save training data
    np.save(os.path.join(os.path.dirname(current_dir), 'results/models/notification_scheduling_q_rewards.npy'), q_rewards)
    np.save(os.path.join(os.path.dirname(current_dir), 'results/models/notification_scheduling_dqn_rewards.npy'), dqn_rewards)
    
    # Test with specific scenarios
    print("\n🧪 Testing specific scenarios...")
    test_notification_scenarios(env, q_agent, dqn_agent, user_profiles)
    
    print("\n✅ Intelligent Notification Scheduling System Training Complete!")
    
    return {
        'q_agent': q_agent,
        'dqn_agent': dqn_agent,
        'environment': env,
        'q_eval': q_eval,
        'dqn_eval': dqn_eval
    }

def test_notification_scenarios(env, q_agent, dqn_agent, user_profiles):
    """Test agents with specific notification scenarios"""
    
    print("\n🎯 Testing specific notification scenarios...")
    
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
        
        for step in range(24):  # One day
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
        
        for step in range(24):  # One day
            action = dqn_agent.choose_action(state, training=False)
            dqn_actions.append(action)
            state, reward, done, info = env.step(action)
            dqn_total_reward += reward
            
            if done:
                break
        
        print(f"   Q-Learning: Total Reward = {q_total_reward:.2f}")
        print(f"   DQN: Total Reward = {dqn_total_reward:.2f}")
        print(f"   Final Notification Summary: {env.get_notification_summary()}")

def analyze_notification_patterns(env, agent, num_episodes=50):
    """Analyze notification scheduling patterns"""
    print("\n📊 Analyzing notification scheduling patterns...")
    
    all_summaries = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state, training=False)
            state, reward, done, info = env.step(action)
        
        summary = env.get_notification_summary()
        all_summaries.append(summary)
    
    # Calculate averages
    avg_delivered = np.mean([s['total_delivered'] for s in all_summaries])
    avg_engagement = np.mean([s['avg_engagement'] for s in all_summaries])
    avg_pending = np.mean([s['pending_count'] for s in all_summaries])
    
    print(f"📈 Average Notifications Delivered: {avg_delivered:.2f}")
    print(f"📈 Average Engagement Rate: {avg_engagement:.3f}")
    print(f"📈 Average Pending Notifications: {avg_pending:.2f}")
    
    # Engagement by type
    engagement_by_type = {}
    for notif_type in env.notification_types:
        engagements = [s['engagement_by_type'].get(notif_type, 0) for s in all_summaries]
        engagement_by_type[notif_type] = np.mean(engagements)
    
    print("\n📊 Engagement by Notification Type:")
    for notif_type, engagement in engagement_by_type.items():
        print(f"   {notif_type}: {engagement:.3f}")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    
    # Run training
    results = train_notification_scheduling()
    
    # Additional analysis
    analyze_notification_patterns(results['environment'], results['dqn_agent'])
