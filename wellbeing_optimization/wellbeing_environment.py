import numpy as np
import random
from typing import Tuple, Dict, Any, List
from gymnasium import spaces

class DigitalWellbeingOptimizationEnvironment:
    """
    Environment for Digital Wellbeing Optimization System
    State: Comprehensive user wellbeing metrics and digital habits
    Actions: Holistic wellbeing interventions and recommendations
    Reward: Based on overall wellbeing improvement and balance
    """
    
    def __init__(self, user_profiles: Dict[int, Dict[str, Any]], 
                 episode_length: int = 7):  # One week
        
        self.user_profiles = user_profiles
        self.episode_length = episode_length
        
        # Wellbeing dimensions
        self.wellbeing_dimensions = ['physical', 'mental', 'social', 'productivity', 'digital_balance']
        self.num_dimensions = len(self.wellbeing_dimensions)
        
        # Intervention types
        self.intervention_types = [
            'screen_time_limit', 'digital_detox', 'mindfulness_reminder', 
            'exercise_prompt', 'social_connection', 'productivity_boost',
            'sleep_optimization', 'battery_conservation', 'app_rearrangement',
            'notification_filter'
        ]
        self.num_interventions = len(self.intervention_types)
        
        # Action space: Choose intervention or no action
        self.action_space = spaces.Discrete(self.num_interventions + 1)  # +1 for no action
        
        # State space: [time, wellbeing_scores, usage_patterns, context, goals]
        self.state_size = 8 + self.num_dimensions * 2 + 5  # Time + wellbeing + usage + context + goals
        
        # Environment state
        self.current_user_id = None
        self.current_day = 0
        self.current_hour = 0
        
        # Wellbeing metrics
        self.wellbeing_scores = {dim: 0.5 for dim in self.wellbeing_dimensions}
        self.wellbeing_history = []
        
        # Digital usage patterns
        self.usage_metrics = {
            'screen_time': 0.0,
            'app_switches': 0,
            'notification_interruptions': 0,
            'deep_work_sessions': 0,
            'social_media_time': 0.0,
            'productive_time': 0.0,
            'entertainment_time': 0.0
        }
        
        # Context and goals
        self.user_context = 'neutral'
        self.wellbeing_goals = {dim: 0.7 for dim in self.wellbeing_dimensions}
        self.intervention_history = []
        self.intervention_effectiveness = {}
        
        # User patterns
        self.wellbeing_patterns = self._generate_wellbeing_patterns()
        self.usage_patterns = self._generate_usage_patterns()
        
    def _generate_wellbeing_patterns(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Generate realistic wellbeing patterns for each user"""
        patterns = {}
        
        for user_id, profile in self.user_profiles.items():
            pattern = {}
            
            # Base wellbeing levels based on user profile
            base_scores = {}
            if profile['usage_pattern'] == 'social_media_heavy':
                base_scores = {
                    'physical': 0.4,  # Less exercise
                    'mental': 0.3,    # Higher stress
                    'social': 0.8,    # More social connection
                    'productivity': 0.4,  # Less productive
                    'digital_balance': 0.2  # Poor balance
                }
            elif profile['usage_pattern'] == 'gaming_heavy':
                base_scores = {
                    'physical': 0.3,  # Much less exercise
                    'mental': 0.4,    # Moderate stress
                    'social': 0.6,    # Some social connection
                    'productivity': 0.3,  # Less productive
                    'digital_balance': 0.3
                }
            elif profile['usage_pattern'] == 'productive':
                base_scores = {
                    'physical': 0.6,  # Moderate exercise
                    'mental': 0.6,    # Good mental state
                    'social': 0.5,    # Balanced social
                    'productivity': 0.8,  # High productivity
                    'digital_balance': 0.7  # Good balance
                }
            else:  # balanced
                base_scores = {
                    'physical': 0.5,
                    'mental': 0.5,
                    'social': 0.5,
                    'productivity': 0.5,
                    'digital_balance': 0.5
                }
            
            # Generate daily patterns with some variation
            daily_patterns = {}
            for day in range(7):
                day_scores = {}
                for dim, base_score in base_scores.items():
                    # Add daily variation
                    variation = random.uniform(-0.1, 0.1)
                    
                    # Weekend effects
                    if day >= 5:  # Weekend
                        if dim == 'social':
                            variation += 0.1  # More social on weekends
                        elif dim == 'productivity':
                            variation -= 0.1  # Less productive on weekends
                        elif dim == 'physical':
                            variation += 0.05  # More exercise on weekends
                    
                    day_scores[dim] = np.clip(base_score + variation, 0.0, 1.0)
                
                daily_patterns[day] = day_scores
            
            pattern['daily'] = daily_patterns
            pattern['base'] = base_scores
            
            patterns[user_id] = pattern
        
        return patterns
    
    def _generate_usage_patterns(self) -> Dict[int, Dict[str, Any]]:
        """Generate digital usage patterns for each user"""
        patterns = {}
        
        for user_id, profile in self.user_profiles.items():
            pattern = {}
            
            # Screen time patterns
            if profile['screen_time_category'] == 'high':
                daily_screen_time = random.uniform(8, 12)
            elif profile['screen_time_category'] == 'moderate':
                daily_screen_time = random.uniform(4, 8)
            else:  # low
                daily_screen_time = random.uniform(1, 4)
            
            # Hourly distribution
            hourly_distribution = np.array([
                0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8,  # 0-7
                1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7,  # 8-15
                0.8, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.2   # 16-23
            ])
            
            # Normalize
            hourly_distribution = hourly_distribution / hourly_distribution.sum()
            
            pattern['daily_screen_time'] = daily_screen_time
            pattern['hourly_distribution'] = hourly_distribution
            
            # App category distribution
            if profile['usage_pattern'] == 'social_media_heavy':
                app_distribution = {'social': 0.5, 'productivity': 0.2, 'entertainment': 0.3}
            elif profile['usage_pattern'] == 'gaming_heavy':
                app_distribution = {'social': 0.2, 'productivity': 0.1, 'entertainment': 0.7}
            elif profile['usage_pattern'] == 'productive':
                app_distribution = {'social': 0.2, 'productivity': 0.6, 'entertainment': 0.2}
            else:
                app_distribution = {'social': 0.3, 'productivity': 0.4, 'entertainment': 0.3}
            
            pattern['app_distribution'] = app_distribution
            
            patterns[user_id] = pattern
        
        return patterns
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        # Select random user
        self.current_user_id = random.choice(list(self.user_profiles.keys()))
        
        # Reset time
        self.current_day = 0
        self.current_hour = 0
        
        # Reset wellbeing scores
        pattern = self.wellbeing_patterns[self.current_user_id]
        self.wellbeing_scores = pattern['daily'][0].copy()
        self.wellbeing_history = [self.wellbeing_scores.copy()]
        
        # Reset usage metrics
        self.usage_metrics = {
            'screen_time': 0.0,
            'app_switches': 0,
            'notification_interruptions': 0,
            'deep_work_sessions': 0,
            'social_media_time': 0.0,
            'productive_time': 0.0,
            'entertainment_time': 0.0
        }
        
        # Reset context and goals
        self.user_context = 'neutral'
        self.wellbeing_goals = {dim: 0.7 for dim in self.wellbeing_dimensions}
        self.intervention_history = []
        self.intervention_effectiveness = {interv: 0.5 for interv in self.intervention_types}
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Time information
        state = np.array([
            self.current_day / 7.0,
            self.current_hour / 24.0,
            self.usage_metrics['screen_time'] / 12.0,  # Normalized to 12 hours
            self.usage_metrics['app_switches'] / 100.0,
            self.usage_metrics['notification_interruptions'] / 50.0,
            self.usage_metrics['deep_work_sessions'] / 10.0,
            self._context_to_numeric(self.user_context),
            len(self.intervention_history) / 20.0
        ])
        
        # Current wellbeing scores
        for dim in self.wellbeing_dimensions:
            state = np.append(state, self.wellbeing_scores[dim])
        
        # Wellbeing goals
        for dim in self.wellbeing_dimensions:
            state = np.append(state, self.wellbeing_goals[dim])
        
        # Recent intervention effectiveness (last 5)
        recent_effectiveness = np.zeros(5)
        if self.intervention_history:
            for i in range(min(5, len(self.intervention_history))):
                intervention = self.intervention_history[-(i+1)]
                effectiveness = self.intervention_effectiveness.get(intervention, 0.5)
                recent_effectiveness[i] = effectiveness
        
        state = np.concatenate([state, recent_effectiveness])
        
        return state
    
    def _context_to_numeric(self, context: str) -> float:
        """Convert context to numeric representation"""
        context_map = {
            'stressed': 0.0,
            'tired': 0.25,
            'neutral': 0.5,
            'focused': 0.75,
            'energetic': 1.0
        }
        return context_map.get(context, 0.5)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time step"""
        # Update usage patterns for this hour
        self._update_usage_patterns()
        
        # Apply intervention if action is not "no action"
        intervention_effect = 0.0
        if action < self.num_interventions:
            intervention = self.intervention_types[action]
            intervention_effect = self._apply_intervention(intervention)
            self.intervention_history.append(intervention)
        
        # Update wellbeing based on usage and interventions
        self._update_wellbeing(intervention_effect)
        
        # Calculate reward
        reward = self._calculate_reward(intervention_effect)
        
        # Update context
        self._update_context()
        
        # Advance time
        self.current_hour += 1
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_day += 1
            
            # Update daily wellbeing pattern
            if self.current_day < 7:
                pattern = self.wellbeing_patterns[self.current_user_id]
                self.wellbeing_scores = pattern['daily'][self.current_day].copy()
        
        # Check if episode is done
        done = (self.current_day >= self.episode_length)
        
        # Store wellbeing history
        self.wellbeing_history.append(self.wellbeing_scores.copy())
        
        # Prepare info dictionary
        info = {
            'user_id': self.current_user_id,
            'day': self.current_day,
            'hour': self.current_hour,
            'wellbeing_scores': self.wellbeing_scores.copy(),
            'usage_metrics': self.usage_metrics.copy(),
            'context': self.user_context,
            'intervention': self.intervention_types[action] if action < self.num_interventions else 'none',
            'intervention_effect': intervention_effect
        }
        
        return self._get_state(), reward, done, info
    
    def _update_usage_patterns(self):
        """Update usage metrics for current hour"""
        usage_pattern = self.usage_patterns[self.current_user_id]
        
        # Hourly screen time
        hourly_screen_time = (usage_pattern['daily_screen_time'] * 
                             usage_pattern['hourly_distribution'][self.current_hour])
        
        self.usage_metrics['screen_time'] += hourly_screen_time
        
        # App-specific usage
        app_dist = usage_pattern['app_distribution']
        self.usage_metrics['social_media_time'] += hourly_screen_time * app_dist['social']
        self.usage_metrics['productive_time'] += hourly_screen_time * app_dist['productivity']
        self.usage_metrics['entertainment_time'] += hourly_screen_time * app_dist['entertainment']
        
        # Other metrics
        self.usage_metrics['app_switches'] += np.random.poisson(hourly_screen_time * 2)
        self.usage_metrics['notification_interruptions'] += np.random.poisson(hourly_screen_time * 1.5)
        
        # Deep work sessions (more likely during work hours)
        if 9 <= self.current_hour <= 17:
            self.usage_metrics['deep_work_sessions'] += np.random.binomial(1, 0.3)
    
    def _apply_intervention(self, intervention: str) -> float:
        """Apply intervention and return its effectiveness"""
        base_effectiveness = self.intervention_effectiveness[intervention]
        
        # Context-dependent effectiveness
        context_modifier = 1.0
        
        if intervention == 'screen_time_limit':
            if self.usage_metrics['screen_time'] > 8:
                context_modifier = 1.2
            elif self.user_context == 'focused':
                context_modifier = 0.8
        
        elif intervention == 'mindfulness_reminder':
            if self.user_context in ['stressed', 'tired']:
                context_modifier = 1.5
            elif self.user_context == 'focused':
                context_modifier = 0.7
        
        elif intervention == 'exercise_prompt':
            if self.wellbeing_scores['physical'] < 0.4:
                context_modifier = 1.3
            elif self.current_hour < 6 or self.current_hour > 20:
                context_modifier = 0.6
        
        elif intervention == 'productivity_boost':
            if self.user_context == 'focused' and 9 <= self.current_hour <= 17:
                context_modifier = 1.4
            elif self.user_context == 'tired':
                context_modifier = 0.5
        
        effectiveness = base_effectiveness * context_modifier
        
        # Update intervention effectiveness based on results (learning)
        if random.random() < 0.1:  # 10% chance to update
            adjustment = random.uniform(-0.05, 0.05)
            self.intervention_effectiveness[intervention] = np.clip(
                effectiveness + adjustment, 0.1, 1.0
            )
        
        return effectiveness
    
    def _update_wellbeing(self, intervention_effect: float):
        """Update wellbeing scores based on usage and interventions"""
        # Base drift towards pattern
        if self.current_day < 7:
            pattern = self.wellbeing_patterns[self.current_user_id]
            target_scores = pattern['daily'][self.current_day]
            
            for dim in self.wellbeing_dimensions:
                # Gradual drift towards pattern
                drift = (target_scores[dim] - self.wellbeing_scores[dim]) * 0.1
                self.wellbeing_scores[dim] += drift
        
        # Usage impacts
        if self.usage_metrics['screen_time'] > 10:
            self.wellbeing_scores['physical'] -= 0.02
            self.wellbeing_scores['mental'] -= 0.01
            self.wellbeing_scores['digital_balance'] -= 0.03
        
        if self.usage_metrics['productive_time'] > 4:
            self.wellbeing_scores['productivity'] += 0.02
            self.wellbeing_scores['digital_balance'] += 0.01
        
        if self.usage_metrics['social_media_time'] > 3:
            self.wellbeing_scores['social'] += 0.01
            self.wellbeing_scores['mental'] -= 0.01
        
        if self.usage_metrics['deep_work_sessions'] > 3:
            self.wellbeing_scores['productivity'] += 0.03
            self.wellbeing_scores['mental'] += 0.01
        
        # Intervention impacts
        if intervention_effect > 0:
            # Positive intervention effect
            for dim in self.wellbeing_dimensions:
                improvement = intervention_effect * 0.02 * random.uniform(0.5, 1.5)
                self.wellbeing_scores[dim] = min(1.0, self.wellbeing_scores[dim] + improvement)
        
        # Ensure scores stay in valid range
        for dim in self.wellbeing_dimensions:
            self.wellbeing_scores[dim] = np.clip(self.wellbeing_scores[dim], 0.0, 1.0)
    
    def _update_context(self):
        """Update user context based on current state"""
        # Determine context based on wellbeing and usage
        avg_wellbeing = np.mean(list(self.wellbeing_scores.values()))
        
        if avg_wellbeing < 0.3:
            self.user_context = 'stressed'
        elif self.wellbeing_scores['physical'] < 0.3:
            self.user_context = 'tired'
        elif avg_wellbeing > 0.7 and self.wellbeing_scores['productivity'] > 0.7:
            self.user_context = 'focused'
        elif avg_wellbeing > 0.6:
            self.user_context = 'energetic'
        else:
            self.user_context = 'neutral'
    
    def _calculate_reward(self, intervention_effect: float) -> float:
        """Calculate reward based on wellbeing improvement"""
        reward = 0.0
        
        # Wellbeing improvement rewards
        for dim in self.wellbeing_dimensions:
            if len(self.wellbeing_history) > 1:
                prev_score = self.wellbeing_history[-2][dim]
                current_score = self.wellbeing_scores[dim]
                improvement = current_score - prev_score
                reward += improvement * 2.0
        
        # Goal achievement rewards
        for dim in self.wellbeing_dimensions:
            goal = self.wellbeing_goals[dim]
            current = self.wellbeing_scores[dim]
            if current >= goal:
                reward += 0.5
            else:
                reward -= 0.1 * (goal - current)
        
        # Balance rewards
        wellbeing_values = list(self.wellbeing_scores.values())
        balance_score = 1.0 - np.std(wellbeing_values)  # Lower std = more balanced
        reward += balance_score * 0.5
        
        # Intervention effectiveness
        if intervention_effect > 0:
            reward += intervention_effect * 0.3
        
        # Usage balance
        total_usage = (self.usage_metrics['social_media_time'] + 
                      self.usage_metrics['productive_time'] + 
                      self.usage_metrics['entertainment_time'])
        
        if total_usage > 0:
            productive_ratio = self.usage_metrics['productive_time'] / total_usage
            if 0.4 <= productive_ratio <= 0.7:  # Good balance
                reward += 0.2
        
        # Penalty for excessive screen time
        if self.usage_metrics['screen_time'] > 12:
            reward -= 0.5
        
        return reward
    
    def get_wellbeing_summary(self) -> Dict[str, Any]:
        """Get comprehensive wellbeing summary"""
        if not self.wellbeing_history:
            return {}
        
        # Calculate trends
        initial_scores = self.wellbeing_history[0]
        current_scores = self.wellbeing_scores
        
        trends = {}
        for dim in self.wellbeing_dimensions:
            trend = current_scores[dim] - initial_scores[dim]
            trends[dim] = trend
        
        # Overall wellbeing score
        overall_score = np.mean(list(current_scores.values()))
        
        # Goal achievement
        goals_achieved = sum(1 for dim in self.wellbeing_dimensions 
                           if current_scores[dim] >= self.wellbeing_goals[dim])
        
        return {
            'overall_score': overall_score,
            'dimension_scores': current_scores.copy(),
            'trends': trends,
            'goals_achieved': goals_achieved,
            'total_goals': len(self.wellbeing_dimensions),
            'usage_metrics': self.usage_metrics.copy(),
            'interventions_used': len(self.intervention_history),
            'most_effective_intervention': max(self.intervention_effectiveness.items(), 
                                               key=lambda x: x[1])[0] if self.intervention_effectiveness else None
        }
