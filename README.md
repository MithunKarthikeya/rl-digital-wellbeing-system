# Reinforcement Learning-Based Adaptive Digital Wellbeing System

A comprehensive RL system that learns to manage digital usage patterns to improve mental and physical wellbeing through intelligent interventions.

## 🎯 Project Overview

This project implements three distinct RL formulations for digital wellbeing management:

1. **App Usage Control System** - Decides when to limit/allow apps based on usage patterns
2. **Intelligent Notification Scheduling System** - Optimizes notification timing to minimize disruption
3. **Digital Wellbeing Optimization System** - Holistic agent balancing screen time, breaks, productivity & mood

Each formulation uses both Q-Learning (tabular baseline) and DQN (neural network-based) algorithms.

## 📊 Dataset

The system uses the `Mobile_Usage_Screentime_Dataset_.xlsx` dataset containing:
- 120 user profiles with comprehensive usage metrics
- Features: Age, Gender, Screen Time, App Usage, Sleep Patterns, Battery Usage
- Realistic usage patterns across different user demographics

## 🏗️ Project Structure

```
rl_new/
├── app_usage_control/           # App Usage Control System
│   ├── environment.py          # RL environment
│   └── train.py               # Training script
├── notification_scheduling/     # Notification Scheduling System
│   ├── environment.py          # RL environment
│   └── train.py               # Training script
├── wellbeing_optimization/     # Digital Wellbeing Optimization System
│   ├── environment.py          # RL environment
│   └── train.py               # Training script
├── utils/                      # Shared utilities
│   ├── data_processor.py       # Data preprocessing
│   ├── rl_algorithms.py       # Q-Learning & DQN implementations
│   └── evaluation.py          # Evaluation & visualization
├── results/                    # Training results and visualizations
├── main.py                    # Main execution script
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rl_new
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run all experiments:
```bash
python main.py
```

### Running Individual Systems

Run specific formulations:
```bash
# App Usage Control only
python main.py --skip-notification --skip-wellbeing

# Notification Scheduling only
python main.py --skip-app-usage --skip-wellbeing

# Digital Wellbeing only
python main.py --skip-app-usage --skip-notification
```

### Quick Testing
For rapid testing with reduced episodes:
```bash
python main.py --quick-run
```

## 🧠 RL Algorithms

### Q-Learning Implementation
- Tabular method with discretized state space
- Epsilon-greedy exploration with decay
- Suitable for simpler state representations
- Fast training and inference

### Deep Q-Network (DQN)
- Neural network function approximation
- Experience replay buffer
- Target network for stable training
- Handles complex, continuous state spaces

## 📱 System Details

### 1. App Usage Control System

**Environment**: Manages app usage across 5 categories (social media, gaming, productivity, entertainment, communication)

**State Space**: Hour of day, usage patterns, battery level, user stress, demographics

**Action Space**: Control decisions (allow/limit/block) for each app category

**Reward Design**: 
- Balanced usage patterns
- Productivity enhancement
- Battery conservation
- Stress reduction

### 2. Intelligent Notification Scheduling System

**Environment**: Optimizes delivery timing for 6 notification types

**State Space**: User activity, focus level, pending notifications, engagement history

**Action Space**: Delivery decisions (now/delay/optimal timing)

**Reward Design**:
- User engagement maximization
- Disruption minimization
- Context-aware timing
- Queue management

### 3. Digital Wellbeing Optimization System

**Environment**: Holistic wellbeing management across 5 dimensions

**State Space**: Wellbeing scores, usage metrics, intervention history, goals

**Action Space**: 10 intervention types + no-action

**Reward Design**:
- Multi-dimensional wellbeing improvement
- Goal achievement
- Balance maintenance
- Long-term sustainability

## 📊 Evaluation Metrics

### Performance Metrics
- **Mean Reward**: Average episode reward
- **Stability**: Inverse of reward variance
- **Learning Efficiency**: Rate of improvement
- **Convergence**: Episode where learning stabilizes

### Visualization Tools
- Training progress curves
- Performance distributions
- Learning phase analysis
- Cross-system comparisons
- Comprehensive dashboards

## 🎨 Results and Visualizations

The system generates comprehensive visualizations:

- **Training Progress**: Learning curves for all agents
- **Performance Analysis**: Distribution comparisons
- **Learning Phases**: Exploration, learning, exploitation phases
- **System Comparison**: Cross-formulation performance analysis
- **Comprehensive Dashboard**: Overview of all metrics

## 📈 Key Findings

### Algorithm Comparison
- **DQN** generally outperforms Q-Learning in complex environments
- **Q-Learning** provides good baseline with lower computational requirements
- Performance varies by system complexity and state space size

### System Performance
1. **Digital Wellbeing Optimization**: Most complex, highest potential impact
2. **App Usage Control**: Consistent performance, practical applications
3. **Notification Scheduling**: Moderate complexity, good user experience impact

## 🔧 Configuration

### Training Parameters
```python
# Q-Learning
learning_rate = 0.1
discount_factor = 0.95
exploration_rate = 1.0
exploration_decay = 0.995

# DQN
learning_rate = 0.001
memory_size = 10000
batch_size = 32
target_update_freq = 1000
```

### Environment Settings
- Episode length: 24 hours × 7 days (one week)
- Evaluation episodes: 100 per agent
- Training episodes: 1000 (Q-Learning), 500 (DQN for wellbeing)

## 🧪 Testing and Validation

### Unit Tests
Each system includes scenario-based testing:
- Different user types (social media heavy, gaming heavy, productive, balanced)
- Edge cases (extreme usage, battery constraints)
- Intervention effectiveness analysis

### Cross-Validation
- Multiple user profiles for generalization
- Different usage patterns validation
- Robustness to noise and variations

## 📚 Dependencies

See `requirements.txt` for complete list:
- Python 3.8+
- PyTorch for neural networks
- NumPy, Pandas for data processing
- Matplotlib, Seaborn for visualization
- Gymnasium for RL environments
- Scikit-learn for preprocessing

## 🔮 Future Work

### Algorithmic Improvements
- Actor-Critic methods (A2C, PPO)
- Multi-agent systems for family/team wellbeing
- Transfer learning across user demographics
- Meta-learning for rapid adaptation

### System Enhancements
- Real-time deployment capabilities
- Mobile application integration
- Personalized recommendation systems
- Longitudinal studies and validation

### Research Directions
- Causal inference for intervention effectiveness
- Privacy-preserving learning methods
- Explainable AI for user trust
- Cross-cultural adaptation studies


---

**Acknowledgments**: This research builds on foundational work in reinforcement learning, digital wellbeing, and human-computer interaction. We thank the research community for valuable insights and methodologies.
