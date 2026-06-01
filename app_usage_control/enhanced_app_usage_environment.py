import numpy as np
import random
from typing import Tuple, Dict, Any
from gymnasium import spaces

class EnhancedAppUsageControlEnvironment:
    """
    Enhanced Environment for App Usage Control System
    Uses multi-dataset integration for richer behavioral modeling
    """
    
    def __init__(self, user_profiles: Dict[int, Dict[str, Any]], 
                 data_processor, max_daily_limit: float = 8.0, episode_length: int = 24):
        
        self.user_profiles = user_profiles
        self.data_processor = data_processor
        self.max_daily_limit = max_daily_limit
        self.episode_length = episode_length
        
        # Define app categories
        self.app_categories = ['social_media', 'gaming', 'productivity', 'entertainment', 'communication']
        self.num_categories = len(self.app_categories)
        
        # Action space: For each category - 0: allow, 1: limit, 2: block
        self.action_space_size = 3 ** self.num_categories
        self.action_space = spaces.Discrete(self.action_space_size)
        
        # Enhanced state space: 20 dimensions
        self.state_size = 20
        
        # Environment state
        self.current_user_id = None
        self.current_hour = 0
        self.current_day = 0
        self.daily_usage = {cat: 0.0 for cat in self.app_categories}
        self.hourly_usage = {cat: [] for cat in self.app_categories}
        self.battery_level = 100.0
        self.user_stress_level = 0.5
        
        # Enhanced behavioral metrics
        self.user_focus_score = 0.5
        self.productivity_score = 0.5
        self.digital_dependence_score = 0.5
        self.addiction_severity = 0
        self.mental_health_composite = 0.5
        
        # Usage patterns based on multi-dataset
        self.usage_patterns = self._generate_enhanced_usage_patterns()
        
    def _generate_enhanced_usage_patterns(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Generate realistic usage patterns using multi-dataset data"""
        patterns = {}
        
        for user_id, profile in self.user_profiles.items():
            pattern = {}
            
            # Base patterns from mobile usage data
            for cat in self.app_categories:
                if profile.get('usage_pattern') == 'social_media_heavy' and cat == 'social_media':
                    # Higher usage during evening hours
                    base_pattern = np.array([0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 1.5, 2.0, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 3.5, 2.8, 2.0, 1.5])
                elif profile.get('usage_pattern') == 'gaming_heavy' and cat == 'gaming':
                    base_pattern = np.array([0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0])
                elif profile.get('usage_pattern') == 'productive' and cat == 'productivity':
                    # Higher usage during work hours
                    base_pattern = np.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3])
                else:
                    # Default balanced pattern
                    base_pattern = np.array([0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3])
                
                # Adjust based on mental health and addiction data
                mental_health_factor = profile.get('stress_level', 0.5) * -1 + 1  # Higher stress = higher usage
                addiction_factor = profile.get('addiction_severity', 0) / 3.0  # Higher addiction = higher usage
                
                pattern[cat] = base_pattern * (1 + mental_health_factor * 0.3 + addiction_factor * 0.5)
                
                # Add some randomness
                noise = np.random.normal(0, 0.1, 24)
                pattern[cat] = np.maximum(0, pattern[cat] + noise)
            
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
        self.battery_level = 100.0
        
        # Reset enhanced behavioral metrics
        profile = self.user_profiles[self.current_user_id]
        self.user_focus_score = profile.get('focus_score', 0.5)
        self.productivity_score = profile.get('productivity_score', 0.5)
        self.digital_dependence_score = profile.get('digital_dependence_score', 0.5)
        self.addiction_severity = profile.get('addiction_severity', 0)
        self.mental_health_composite = (profile.get('stress_level', 0.5) + 
                                   profile.get('happiness_score', 0.5)) / 2
        
        return self._get_enhanced_state()
    
    def _get_enhanced_state(self) -> np.ndarray:
        """Get enhanced state representation with multi-dataset features"""
        profile = self.user_profiles[self.current_user_id]
        
        # Time and context features
        state = np.array([
            self.current_hour / 24.0,  # Normalized hour
            self.current_day / 7.0,    # Normalized day
            self.battery_level / 100.0,  # Normalized battery
            sum(self.daily_usage.values()) / self.max_daily_limit,  # Total usage ratio
        ])
        
        # Enhanced behavioral features
        behavioral_features = np.array([
            self.user_focus_score,      # Focus level
            self.productivity_score,    # Productivity
            self.digital_dependence_score,  # Digital dependence
            self.addiction_severity / 3.0,  # Normalized addiction severity
            self.mental_health_composite,  # Mental health composite
            profile.get('anxiety_score', 0.5) / 10.0,  # Normalized anxiety
            profile.get('depression_score', 0.5) / 10.0,  # Normalized depression
            profile.get('happiness_score', 0.5) / 10.0,  # Normalized happiness
        ])
        
        # Usage ratios for each category
        total_usage = sum(self.daily_usage.values())
        if total_usage > 0:
            usage_ratios = np.array([
                self.daily_usage['social_media'] / total_usage,
                self.daily_usage['gaming'] / total_usage,
                self.daily_usage['productivity'] / total_usage,
                self.daily_usage['entertainment'] / total_usage,
                self.daily_usage['communication'] / total_usage
            ])
        else:
            usage_ratios = np.zeros(5)
        
        # Device and demographic features
        device_features = np.array([
            1.0 if profile.get('device_type') == 'iPhone' else 0.0,  # Device type
            profile.get('age', 25) / 50.0,  # Normalized age
            1.0 if profile.get('gender') == 'Male' else 0.0,  # Gender
            1.0 if profile.get('high_risk_flag', False) else 0.0,  # Risk flag
        ])
        
        # Combine all features
        state = np.concatenate([state, behavioral_features, usage_ratios, device_features])
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time step"""
        # Decode action
        control_decisions = self._decode_action(action)
        
        # Generate usage for this hour
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
            
            hour_usage[cat] = max(0, actual_usage)
            self.daily_usage[cat] += actual_usage
            self.hourly_usage[cat].append(actual_usage)
        
        # Update battery level
        total_hour_usage = sum(hour_usage.values())
        self.battery_level -= total_hour_usage * 2  # 2% battery drain per hour of usage
        self.battery_level = max(0, self.battery_level)
        
        # Update enhanced behavioral metrics
        self._update_enhanced_metrics(control_decisions, hour_usage)
        
        # Calculate reward
        reward = self._calculate_enhanced_reward(control_decisions, hour_usage)
        
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
            'day': self.current_day,
            'daily_usage': dict(self.daily_usage),
            'battery_level': self.battery_level,
            'control_decisions': control_decisions,
            'focus_score': self.user_focus_score,
            'productivity_score': self.productivity_score,
            'addiction_severity': self.addiction_severity
        }
        
        return self._get_enhanced_state(), reward, done, info
    
    def _decode_action(self, action: int) -> Dict[str, int]:
        """Convert discrete action to category-specific control decisions"""
        control_decisions = {}
        remaining_action = action
        
        for cat in self.app_categories:
            control_decisions[cat] = remaining_action % 3
            remaining_action //= 3
        
        return control_decisions
    
    def _update_enhanced_metrics(self, control_decisions: Dict[str, int], hour_usage: Dict[str, float]):
        """Update enhanced behavioral metrics based on current actions"""
        # Update focus based on control strictness
        strict_control_count = sum(1 for decision in control_decisions.values() if decision == 2)  # Block
        if strict_control_count >= 3:
            self.user_focus_score = max(0.1, self.user_focus_score - 0.05)
        elif strict_control_count <= 1:
            self.user_focus_score = min(1.0, self.user_focus_score + 0.02)
        
        # Update productivity based on productive app usage
        productive_usage = hour_usage.get('productivity', 0)
        total_usage = sum(hour_usage.values())
        if total_usage > 0:
            productivity_ratio = productive_usage / total_usage
            if productivity_ratio > 0.5:
                self.productivity_score = min(1.0, self.productivity_score + 0.03)
            else:
                self.productivity_score = max(0.2, self.productivity_score - 0.02)
        
        # Update digital dependence based on social media and entertainment usage
        distracting_usage = hour_usage.get('social_media', 0) + hour_usage.get('entertainment', 0)
        if total_usage > 0:
            dependence_ratio = distracting_usage / total_usage
            if dependence_ratio > 0.6:
                self.digital_dependence_score = min(1.0, self.digital_dependence_score + 0.04)
            else:
                self.digital_dependence_score = max(0.1, self.digital_dependence_score - 0.02)
    
    def _calculate_enhanced_reward(self, control_decisions: Dict[str, int], hour_usage: Dict[str, float]) -> float:
        """Calculate reward using enhanced behavioral modeling"""
        reward = 0.0
        profile = self.user_profiles[self.current_user_id]
        
        # Base usage balance reward
        total_usage = sum(hour_usage.values())
        if 2.0 <= total_usage <= 6.0:  # Optimal usage range
            reward += 2.0
        elif total_usage > 10.0:  # Excessive usage penalty
            reward -= 1.0
        
        # Productivity enhancement reward
        productive_usage = hour_usage.get('productivity', 0)
        if total_usage > 0:
            productivity_ratio = productive_usage / total_usage
            reward += productivity_ratio * 3.0
        
        # Mental health consideration
        mental_health_bonus = (self.mental_health_composite - 0.5) * 2.0  # Positive mental health bonus
        reward += mental_health_bonus
        
        # Addiction-aware control rewards
        if self.addiction_severity >= 2:  # High addiction
            # Reward strict control for addicted users
            strict_count = sum(1 for decision in control_decisions.values() if decision == 2)
            reward += strict_count * 0.5
        elif self.addiction_severity == 0:  # No addiction
            # Reward lenient control for healthy users
            lenient_count = sum(1 for decision in control_decisions.values() if decision == 0)
            reward += lenient_count * 0.2
        
        # Focus preservation reward
        if self.user_focus_score > 0.7:
            focus_reward = 1.0
        elif self.user_focus_score < 0.3:
            focus_reward = -0.5
        else:
            focus_reward = 0.0
        reward += focus_reward
        
        # Battery conservation bonus
        if self.battery_level > 50:
            reward += 0.5
        
        # Personalization based on user profile
        if profile.get('usage_pattern') == 'productive':
            if control_decisions.get('productivity', 0) == 0:  # Allow productivity
                reward += 1.0
        elif profile.get('usage_pattern') == 'social_media_heavy':
            if control_decisions.get('social_media', 0) == 1:  # Limit social media
                reward += 1.0
        
        return reward
    
    def get_enhanced_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary with enhanced metrics"""
        return {
            'daily_usage': dict(self.daily_usage),
            'total_usage': sum(self.daily_usage.values()),
            'battery_level': self.battery_level,
            'user_id': self.current_user_id,
            'focus_score': self.user_focus_score,
            'productivity_score': self.productivity_score,
            'digital_dependence_score': self.digital_dependence_score,
            'addiction_severity': self.addiction_severity,
            'mental_health_composite': self.mental_health_composite,
            'control_decisions': self._decode_action(0)  # Get last action
        }
