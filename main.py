#!/usr/bin/env python3
"""
Reinforcement Learning-Based Adaptive Digital Wellbeing System
Main script to run all experiments and evaluations

This system implements three RL formulations:
1. App Usage Control System
2. Intelligent Notification Scheduling System  
3. Digital Wellbeing Optimization System

Each formulation uses both Q-Learning and DQN algorithms.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.data_processor import DataProcessor
from utils.evaluation import RLEvaluator, Visualizer
from app_usage_control.train import train_app_usage_control
from notification_scheduling.train import train_notification_scheduling
from wellbeing_optimization.train import train_wellbeing_optimization

def setup_directories():
    """Create necessary directories for results"""
    directories = [
        'results', 'results/models', 'results/plots', 'results/reports', 
        'results/data', 'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("📁 Directory structure created")

def run_data_exploration():
    """Explore and visualize the dataset"""
    print("\n" + "="*60)
    print("🔍 DATA EXPLORATION AND ANALYSIS")
    print("="*60)
    
    # Initialize data processor
    data_processor = DataProcessor('Mobile_Usage_Screentime_Dataset_.xlsx')
    
    # Load and explore data
    exploration_results = data_processor.explore_data()
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Shape: {exploration_results['shape']}")
    print(f"   Columns: {len(exploration_results['columns'])}")
    print(f"   Missing values: {sum(exploration_results['missing_values'].values())}")
    
    # Create user profiles
    user_profiles = data_processor.create_user_profiles()
    
    print(f"\n👥 User Profile Analysis:")
    usage_patterns = {}
    for user_id, profile in user_profiles.items():
        pattern = profile['usage_pattern']
        if pattern not in usage_patterns:
            usage_patterns[pattern] = 0
        usage_patterns[pattern] += 1
    
    for pattern, count in usage_patterns.items():
        print(f"   {pattern.replace('_', ' ').title()}: {count} users")
    
    # Visualize data
    data_processor.visualize_data('results/data/data_exploration.png')
    
    return data_processor, user_profiles

def run_all_experiments(args):
    """Run all three RL formulations with both algorithms"""
    
    print("\n" + "="*60)
    print("🚀 STARTING REINFORCEMENT LEARNING EXPERIMENTS")
    print("="*60)
    
    # Track all results
    all_results = {}
    training_data = {}
    evaluation_data = {}
    
    # 1. App Usage Control System
    if not args.skip_app_usage:
        print("\n📱 Running App Usage Control System experiments...")
        app_usage_results = train_app_usage_control()
        all_results['app_usage_control'] = app_usage_results
        
        # Collect training data
        training_data['App Usage Q-Learning'] = app_usage_results['q_agent'].episode_rewards
        training_data['App Usage DQN'] = app_usage_results['dqn_agent'].episode_rewards
        
        print("✅ App Usage Control System completed")
    
    # 2. Notification Scheduling System
    if not args.skip_notification:
        print("\n🔔 Running Intelligent Notification Scheduling System experiments...")
        notification_results = train_notification_scheduling()
        all_results['notification_scheduling'] = notification_results
        
        # Collect training data
        training_data['Notification Q-Learning'] = notification_results['q_agent'].episode_rewards
        training_data['Notification DQN'] = notification_results['dqn_agent'].episode_rewards
        
        print("✅ Intelligent Notification Scheduling System completed")
    
    # 3. Digital Wellbeing Optimization System
    if not args.skip_wellbeing:
        print("\n🧘 Running Digital Wellbeing Optimization System experiments...")
        wellbeing_results = train_wellbeing_optimization()
        all_results['wellbeing_optimization'] = wellbeing_results
        
        # Collect training data
        training_data['Wellbeing Q-Learning'] = wellbeing_results['q_agent'].episode_rewards
        training_data['Wellbeing DQN'] = wellbeing_results['dqn_agent'].episode_rewards
        
        print("✅ Digital Wellbeing Optimization System completed")
    
    return all_results, training_data

def comprehensive_evaluation(all_results: Dict[str, Any], training_data: Dict[str, List[float]]):
    """Perform comprehensive evaluation across all systems"""
    
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE EVALUATION AND ANALYSIS")
    print("="*60)
    
    # Initialize evaluator and visualizer
    evaluator = RLEvaluator('results')
    visualizer = Visualizer('results')
    
    # Analyze training curves
    print("\n📈 Analyzing training curves...")
    training_analysis = evaluator.analyze_training_curves(training_data)
    
    # Generate comprehensive visualizations
    print("\n🎨 Generating comprehensive visualizations...")
    
    # 1. Training comparison across all agents
    visualizer.plot_training_comparison(
        training_data, 
        'results/plots/all_agents_training_comparison.png'
    )
    
    # 2. Learning phases analysis
    visualizer.plot_learning_phases(
        training_data,
        'results/plots/learning_phases_analysis.png'
    )
    
    # 3. Performance evaluation for each system
    for system_name, system_results in all_results.items():
        print(f"\n🔍 Evaluating {system_name.replace('_', ' ').title()}...")
        
        # Prepare evaluation data
        eval_data = {
            'Q-Learning': system_results['q_eval'],
            'DQN': system_results['dqn_eval']
        }
        
        # Add raw rewards if available
        if hasattr(system_results['q_agent'], 'episode_rewards'):
            eval_data['Q-Learning']['rewards'] = system_results['q_agent'].episode_rewards[-100:]  # Last 100 episodes
        if hasattr(system_results['dqn_agent'], 'episode_rewards'):
            eval_data['DQN']['rewards'] = system_results['dqn_agent'].episode_rewards[-100:]  # Last 100 episodes
        
        # Calculate additional metrics using evaluator
        for agent_name in ['Q-Learning', 'DQN']:
            if 'rewards' in eval_data[agent_name]:
                rewards = eval_data[agent_name]['rewards']
                eval_data[agent_name]['stability'] = evaluator._calculate_stability(rewards)
                eval_data[agent_name]['learning_efficiency'] = evaluator._calculate_learning_efficiency(rewards)
                eval_data[agent_name]['convergence_episode'] = evaluator._find_convergence_episode(rewards)
        
        # Plot performance distribution
        visualizer.plot_performance_distribution(
            eval_data,
            f'results/plots/{system_name}_performance_distribution.png'
        )
        
        # Generate evaluation report
        report = evaluator.generate_evaluation_report(
            eval_data,
            f'results/reports/{system_name}_evaluation_report.md'
        )
    
    # 4. Cross-system comparison
    print("\n🔄 Performing cross-system comparison...")
    
    # Collect all DQN results for comparison
    cross_system_data = {}
    for system_name, system_results in all_results.items():
        agent_name = f"{system_name.replace('_', ' ').title()} DQN"
        cross_system_data[agent_name] = system_results['dqn_eval']
        
        # Add rewards if available
        if hasattr(system_results['dqn_agent'], 'episode_rewards'):
            cross_system_data[agent_name]['rewards'] = system_results['dqn_agent'].episode_rewards[-100:]
            
            # Calculate additional metrics
            rewards = cross_system_data[agent_name]['rewards']
            cross_system_data[agent_name]['stability'] = evaluator._calculate_stability(rewards)
            cross_system_data[agent_name]['learning_efficiency'] = evaluator._calculate_learning_efficiency(rewards)
            cross_system_data[agent_name]['convergence_episode'] = evaluator._find_convergence_episode(rewards)
    
    # Plot cross-system comparison
    visualizer.plot_performance_distribution(
        cross_system_data,
        'results/plots/cross_system_comparison.png'
    )
    
    # 5. Create comprehensive dashboard
    print("\n📊 Creating comprehensive evaluation dashboard...")
    
    # Collect training data for dashboard
    dashboard_training_data = {}
    for name, rewards in training_data.items():
        if 'DQN' in name:  # Focus on DQN for dashboard
            dashboard_training_data[name] = rewards
    
    # Prepare evaluation data for dashboard
    dashboard_eval_data = {}
    for system_name, system_results in all_results.items():
        agent_name = f"{system_name.replace('_', ' ').title()} DQN"
        dashboard_eval_data[agent_name] = system_results['dqn_eval']
        
        # Add rewards
        if hasattr(system_results['dqn_agent'], 'episode_rewards'):
            dashboard_eval_data[agent_name]['rewards'] = system_results['dqn_agent'].episode_rewards
            
            # Calculate additional metrics
            rewards = dashboard_eval_data[agent_name]['rewards']
            dashboard_eval_data[agent_name]['stability'] = evaluator._calculate_stability(rewards)
            dashboard_eval_data[agent_name]['learning_efficiency'] = evaluator._calculate_learning_efficiency(rewards)
            dashboard_eval_data[agent_name]['convergence_episode'] = evaluator._find_convergence_episode(rewards)
    
    visualizer.create_comprehensive_dashboard(
        dashboard_training_data,
        dashboard_eval_data,
        'results/plots/comprehensive_dashboard.png'
    )
    
    # 6. Generate final summary report
    print("\n📝 Generating final summary report...")
    generate_final_summary_report(all_results, training_data, training_analysis)
    
    print("\n✅ Comprehensive evaluation completed!")

def generate_final_summary_report(all_results: Dict[str, Any], 
                                training_data: Dict[str, List[float]],
                                training_analysis: Dict[str, Any]):
    """Generate a final summary report with key findings"""
    
    report = """# Reinforcement Learning-Based Adaptive Digital Wellbeing System
## Final Evaluation Report

### Executive Summary

This report presents the comprehensive evaluation of a Reinforcement Learning-Based Adaptive Digital Wellbeing System, implementing three distinct formulations:

1. **App Usage Control System** - Intelligent management of application usage patterns
2. **Intelligent Notification Scheduling System** - Optimal timing for notification delivery
3. **Digital Wellbeing Optimization System** - Holistic wellbeing management

Each formulation was implemented using both Q-Learning (tabular) and Deep Q-Network (DQN) algorithms.

"""
    
    # System performance summary
    report += "### System Performance Summary\n\n"
    report += "| System | Algorithm | Mean Reward | Stability | Learning Efficiency |\n"
    report += "|--------|-----------|-------------|-----------|---------------------|\n"
    
    for system_name, system_results in all_results.items():
        system_title = system_name.replace('_', ' ').title()
        
        # Q-Learning results
        q_results = system_results['q_eval']
        report += f"| {system_title} | Q-Learning | {q_results['mean_reward']:.3f} | "
        
        # Calculate stability if not available
        if 'stability' in q_results:
            report += f"{q_results['stability']:.3f} | "
        else:
            report += "N/A | "
        
        if 'learning_efficiency' in q_results:
            report += f"{q_results['learning_efficiency']:.3f} |\n"
        else:
            report += "N/A |\n"
        
        # DQN results
        dqn_results = system_results['dqn_eval']
        report += f"| {system_title} | DQN | {dqn_results['mean_reward']:.3f} | "
        
        if 'stability' in dqn_results:
            report += f"{dqn_results['stability']:.3f} | "
        else:
            report += "N/A | "
        
        if 'learning_efficiency' in dqn_results:
            report += f"{dqn_results['learning_efficiency']:.3f} |\n"
        else:
            report += "N/A |\n"
    
    # Key findings
    report += "\n### Key Findings\n\n"
    
    # Best performing agent
    best_performance = -float('inf')
    best_agent = ""
    
    for system_name, system_results in all_results.items():
        for agent_type, eval_results in [('q_agent', 'Q-Learning'), ('dqn_agent', 'DQN')]:
            agent = system_results[agent_type]
            if hasattr(agent, 'episode_rewards') and agent.episode_rewards:
                avg_reward = np.mean(agent.episode_rewards[-100:])  # Last 100 episodes
                if avg_reward > best_performance:
                    best_performance = avg_reward
                    best_agent = f"{system_name.replace('_', ' ').title()} {eval_results}"
    
    report += f"**Best Performing Agent**: {best_agent} (Average Reward: {best_performance:.3f})\n\n"
    
    # Algorithm comparison
    report += "#### Algorithm Comparison\n\n"
    
    q_total = 0
    dqn_total = 0
    q_count = 0
    dqn_count = 0
    
    for system_results in all_results.values():
        q_total += system_results['q_eval']['mean_reward']
        dqn_total += system_results['dqn_eval']['mean_reward']
        q_count += 1
        dqn_count += 1
    
    q_avg = q_total / q_count
    dqn_avg = dqn_total / dqn_count
    
    if dqn_avg > q_avg:
        report += f"**DQN outperforms Q-Learning** by {dqn_avg - q_avg:.3f} points on average\n"
    else:
        report += f"**Q-Learning outperforms DQN** by {q_avg - dqn_avg:.3f} points on average\n"
    
    # System-specific insights
    report += "\n#### System-Specific Insights\n\n"
    
    for system_name, system_results in all_results.items():
        system_title = system_name.replace('_', ' ').title()
        
        # Determine which algorithm performed better for this system
        q_reward = system_results['q_eval']['mean_reward']
        dqn_reward = system_results['dqn_eval']['mean_reward']
        
        if dqn_reward > q_reward:
            better_algo = "DQN"
            improvement = ((dqn_reward - q_reward) / q_reward) * 100
        else:
            better_algo = "Q-Learning"
            improvement = ((q_reward - dqn_reward) / dqn_reward) * 100
        
        report += f"**{system_title}**: {better_algo} performs better by {improvement:.1f}%\n"
    
    # Recommendations
    report += "\n### Recommendations\n\n"
    report += "Based on the comprehensive evaluation, the following recommendations are made:\n\n"
    
    report += "1. **For Real-time Applications**: Use DQN for its superior performance in complex state spaces\n"
    report += "2. **For Resource-Constrained Environments**: Q-Learning provides good performance with lower computational requirements\n"
    report += "3. **For Personal Digital Wellbeing**: The Digital Wellbeing Optimization System with DQN shows the most promise\n"
    report += "4. **For Enterprise Applications**: The App Usage Control System provides the most consistent performance\n"
    
    # Future work
    report += "\n### Future Work\n\n"
    report += "1. **Multi-Agent Systems**: Implement collaborative agents for family or team wellbeing management\n"
    report += "2. **Transfer Learning**: Apply learned policies across different user demographics\n"
    report += "3. **Real-world Deployment**: Conduct user studies with actual digital wellbeing applications\n"
    report += "4. **Advanced Algorithms**: Explore Actor-Critic methods and Proximal Policy Optimization (PPO)\n"
    
    # Save report
    with open('results/reports/final_summary_report.md', 'w') as f:
        f.write(report)
    
    print("📄 Final summary report saved to 'results/reports/final_summary_report.md'")

def main():
    """Main function to run all experiments"""
    
    parser = argparse.ArgumentParser(description='RL-Based Digital Wellbeing System')
    parser.add_argument('--skip-app-usage', action='store_true', 
                       help='Skip App Usage Control System experiments')
    parser.add_argument('--skip-notification', action='store_true', 
                       help='Skip Notification Scheduling System experiments')
    parser.add_argument('--skip-wellbeing', action='store_true', 
                       help='Skip Digital Wellbeing Optimization System experiments')
    parser.add_argument('--quick-run', action='store_true', 
                       help='Run with reduced episodes for quick testing')
    
    args = parser.parse_args()
    
    print("🎯 Reinforcement Learning-Based Adaptive Digital Wellbeing System")
    print("="*60)
    print("Starting comprehensive experiments...")
    
    start_time = time.time()
    
    try:
        # Setup
        setup_directories()
        
        # Data exploration
        data_processor, user_profiles = run_data_exploration()
        
        # Run all experiments
        all_results, training_data = run_all_experiments(args)
        
        # Comprehensive evaluation
        comprehensive_evaluation(all_results, training_data)
        
        # Final summary
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("🎉 EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️  Total execution time: {duration/60:.1f} minutes")
        print(f"📁 Results saved in 'results/' directory")
        print(f"📊 Key visualizations:")
        print(f"   - Comprehensive dashboard: results/plots/comprehensive_dashboard.png")
        print(f"   - Training comparison: results/plots/all_agents_training_comparison.png")
        print(f"   - Cross-system comparison: results/plots/cross_system_comparison.png")
        print(f"📄 Reports:")
        print(f"   - Final summary: results/reports/final_summary_report.md")
        print(f"   - Individual system reports: results/reports/")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
