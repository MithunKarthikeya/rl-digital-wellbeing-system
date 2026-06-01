import numpy as np
import random
from typing import Tuple, Dict, Any
from gymnasium import spaces

class AppUsageControlEnvironment:
    """
    Environment for App Usage Control System
    State: User's current app usage patterns and context
    Actions: Control decisions for different app categories
    Reward: Based on achieving healthy usage balance
    """
    
    def __init__(self, user_profiles: Dict[int, Dict[str, Any]], 
                 max_daily_limit: float = 8.0, episode_length: int = 24):
        
        self.user_profiles = user_profiles
        self.max_daily_limit = max_daily_limit
        self.episode_length = episode_length
        
        # Define app categories
        self.app_categories = ['social_media', 'gaming', 'productivity', 'entertainment', 'communication']
        self.num_categories = len(self.app_categories)
        
        # Action space: For each category - 0: allow, 1: limit, 2: block
        self.action_space = spaces.Discrete(3 ** self.num_categories)
        
        # State space: [hour_of_day, day_of_week, current_usage_ratios, battery_level, user_stress_level]
        self.state_size = 5 + self.num_categories + 2  # 5 basic + num_categories usage + 2 additional
        
        # Environment state
        self.current_user_id = None
        self.current_hour = 0
        self.current_day = 0
        self.daily_usage = {cat: 0.0 for cat in self.app_categories}
        self.hourly_usage = {cat: [] for cat in self.app_categories}
        self.battery_level = 100.0
        self.user_stress_level = 0.5
        
        # Usage patterns based on user profiles
        self.usage_patterns = self._generate_usage_patterns()
        
    def _generate_usage_patterns(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Generate realistic usage patterns for each user"""
        patterns = {}
        
        for user_id, profile in self.user_profiles.items():
            pattern = {}
            
            # Generate hourly usage patterns for each category
            for cat in self.app_categories:
                if profile['usage_pattern'] == 'social_media_heavy' and cat == 'social_media':
                    # Higher usage during evening hours
                    base_pattern = np.array([0.5, 0.3, 0.4, 0.6, 0.8, 1.2, 1.5, 2.0, 1.8, 2.2, 2.5, 2.8,
                                           3.0, 3.2, 3.5, 3.8, 4.0, 4.5, 4.8, 4.2, 3.5, 2.8, 1.5, 0.8])
                elif profile['usage_pattern'] == 'gaming_heavy' and cat == 'gaming':
                    # Higher usage during late night and weekend
                    base_pattern = np.array([0.2, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8,
                                           2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0, 4.5, 4.8, 4.2, 3.0, 1.5])
                elif profile['usage_pattern'] == 'productive' and cat == 'productivity':
                    # Higher usage during work hours
                    base_pattern = np.array([0.1, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8, 3.0, 2.5, 2.0,
                                           1.8, 1.5, 1.2, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1])
                else:
                    # Default balanced pattern
                    base_pattern = np.random.uniform(0.1, 1.0, 24)
                
                # Add some randomness
                noise = np.random.normal(0, 0.1, 24)
                pattern[cat] = np.maximum(0, base_pattern + noise)
            
            patterns[user_id] = pattern
        
        return patterns
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        # Select random user
        self.current_user_id = random.choice(list(self.user_profiles.keys()))
        
        # Reset time
        self.current_hour = 0
        self.current_day = 0
        
        # Reset usage tracking
        self.daily_usage = {cat: 0.0 for cat in self.app_categories}
        self.hourly_usage = {cat: [] for cat in self.app_categories}
        
        # Reset battery and stress
        self.battery_level = 100.0
        self.user_stress_level = 0.5
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = np.array([
            self.current_hour / 24.0,  # Normalized hour
            self.current_day / 7.0,    # Normalized day of week
            self.battery_level / 100.0,  # Normalized battery
            self.user_stress_level,    # Stress level
            sum(self.daily_usage.values()) / self.max_daily_limit  # Total usage ratio
        ])
        
        # Add usage ratios for each category
        for cat in self.app_categories:
            max_expected = 4.0  # Max 4 hours per category
            state = np.append(state, min(self.daily_usage[cat] / max_expected, 1.0))
        
        # Add user profile features
        profile = self.user_profiles[self.current_user_id]
        state = np.append(state, profile['age'] / 50.0)  # Normalized age
        state = np.append(state, 1.0 if profile['screen_time_category'] == 'high' else 0.0)  # High usage indicator
        
        return state
    
    def _decode_action(self, action: int) -> Dict[str, int]:
        """Convert discrete action to category-specific control decisions"""
        control_decisions = {}
        remaining_action = action
        
        for i, cat in enumerate(self.app_categories):
            control_decisions[cat] = remaining_action % 3
            remaining_action //= 3
        
        return control_decisions
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time step"""
        # Decode action
        control_decisions = self._decode_action(action)
        
        # Get user's natural usage tendency for this hour
        pattern = self.usage_patterns[self.current_user_id]
        hour_usage = {}
        
        for cat in self.app_categories:
            natural_usage = pattern[cat][self.current_hour]
            
            # Apply control decision
            if control_decisions[cat] == 0:  # Allow
                actual_usage = natural_usage
            elif control_decisions[cat] == 1:  # Limit
                actual_usage = natural_usage * 0.5
            else:  # Block
                actual_usage = natural_usage * 0.1
            
            hour_usage[cat] = actual_usage
            self.daily_usage[cat] += actual_usage
            self.hourly_usage[cat].append(actual_usage)
        
        # Update battery level
        total_hour_usage = sum(hour_usage.values())
        self.battery_level -= total_hour_usage * 2  # 2% battery drain per hour of usage
        self.battery_level = max(0, self.battery_level)
        
        # Update stress level based on usage patterns
        self._update_stress_level(hour_usage, control_decisions)
        
        # Calculate reward
        reward = self._calculate_reward(hour_usage, control_decisions)
        
        # Advance time
        self.current_hour += 1
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_day += 1
        
        # Check if episode is done
        done = (self.current_day >= 1) or (self.battery_level <= 10) or (sum(self.daily_usage.values()) >= self.max_daily_limit * 1.5)
        
        # Prepare info dictionary
        info = {
            'user_id': self.current_user_id,
            'hour': self.current_hour,
            'daily_usage': dict(self.daily_usage),
            'battery_level': self.battery_level,
            'stress_level': self.user_stress_level,
            'control_decisions': control_decisions
        }
        
        return self._get_state(), reward, done, info
    
    def _update_stress_level(self, hour_usage: Dict[str, float], control_decisions: Dict[str, int]):
        """Update user stress level based on usage and control"""
        # High social media usage increases stress
        social_stress = hour_usage['social_media'] * 0.1
        
        # Productive usage decreases stress
        productivity_relief = hour_usage['productivity'] * -0.05
        
        # Strict control increases stress
        control_stress = sum(1 for decision in control_decisions.values() if decision == 2) * 0.02
        
        # Update stress level with smoothing
        stress_change = social_stress + productivity_relief + control_stress
        self.user_stress_level = np.clip(self.user_stress_level + stress_change, 0.0, 1.0)
    
    def _calculate_reward(self, hour_usage: Dict[str, float], control_decisions: Dict[str, int]) -> float:
        """Calculate reward based on usage balance and user wellbeing"""
        reward = 0.0
        
        # Reward for balanced usage
        total_usage = sum(hour_usage.values())
        if 1.0 <= total_usage <= 3.0:  # Optimal hourly usage
            reward += 1.0
        elif total_usage > 3.0:
            reward -= 0.5  # Penalty for excessive usage
        else:
            reward += 0.3  # Small reward for low usage
        
        # Reward for productive activities
        productivity_ratio = hour_usage['productivity'] / max(total_usage, 0.1)
        reward += productivity_ratio * 0.5
        
        # Penalty for excessive social media
        social_ratio = hour_usage['social_media'] / max(total_usage, 0.1)
        if social_ratio > 0.6:
            reward -= 0.3
        
        # Reward for appropriate control
        profile = self.user_profiles[self.current_user_id]
        if profile['usage_pattern'] == 'social_media_heavy':
            # Reward limiting social media for heavy users
            if control_decisions['social_media'] == 1:  # Limited
                reward += 0.3
            elif control_decisions['social_media'] == 2:  # Blocked
                reward += 0.1
        
        # Penalty for very strict control (user frustration)
        strict_control_count = sum(1 for decision in control_decisions.values() if decision == 2)
        if strict_control_count >= 3:
            reward -= 0.2
        
        # Battery conservation bonus
        if self.battery_level > 50:
            reward += 0.1
        
        # Stress management bonus
        if self.user_stress_level < 0.3:
            reward += 0.2
        elif self.user_stress_level > 0.7:
            reward -= 0.2
        
        return reward
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of usage patterns"""
        return {
            'daily_usage': dict(self.daily_usage),
            'total_usage': sum(self.daily_usage.values()),
            'battery_level': self.battery_level,
            'stress_level': self.user_stress_level,
            'user_id': self.current_user_id
        }
