import numpy as np
import random
from typing import Tuple, Dict, Any, List
from gymnasium import spaces

class NotificationSchedulingEnvironment:
    """
    Environment for Intelligent Notification Scheduling System
    State: User context, current activity, notification queue
    Actions: When to deliver notifications (delay, deliver now, schedule later)
    Reward: Based on user engagement and disruption minimization
    """
    
    def __init__(self, user_profiles: Dict[int, Dict[str, Any]], 
                 max_notifications_per_hour: int = 10, episode_length: int = 24):
        
        self.user_profiles = user_profiles
        self.max_notifications_per_hour = max_notifications_per_hour
        self.episode_length = episode_length
        
        # Notification categories
        self.notification_types = ['social', 'email', 'messaging', 'system', 'productivity', 'entertainment']
        self.num_types = len(self.notification_types)
        
        # Action space: For each pending notification - 0: deliver now, 1: delay 1 hour, 2: schedule optimal time
        self.action_space = spaces.Discrete(3)  # Simplified to single decision per step
        
        # State space: [hour, day_of_week, user_activity, focus_level, battery, pending_notifications, recent_deliveries]
        self.state_size = 6 + self.num_types + 5  # Basic + notification queue + delivery history
        
        # Environment state
        self.current_user_id = None
        self.current_hour = 0
        self.current_day = 0
        self.user_activity = 'idle'  # idle, working, socializing, sleeping, exercising
        self.focus_level = 0.5
        self.battery_level = 100.0
        
        # Notification management
        self.pending_notifications = []
        self.delivered_notifications = []
        self.notification_queue = []
        self.engagement_history = []
        
        # User activity patterns
        self.activity_patterns = self._generate_activity_patterns()
        self.notification_patterns = self._generate_notification_patterns()
        
    def _generate_activity_patterns(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Generate realistic activity patterns for each user"""
        patterns = {}
        
        for user_id, profile in self.user_profiles.items():
            pattern = {}
            
            # Generate hourly activity patterns
            activities = ['sleeping', 'working', 'socializing', 'exercising', 'idle']
            hourly_activity = []
            
            for hour in range(24):
                if 0 <= hour < 7:  # Night
                    activity = 'sleeping'
                elif 7 <= hour < 9:  # Morning routine
                    activity = random.choice(['idle', 'exercising'])
                elif 9 <= hour < 17:  # Work hours
                    if profile['usage_pattern'] == 'productive':
                        activity = 'working'
                    else:
                        activity = random.choice(['working', 'idle', 'socializing'])
                elif 17 <= hour < 21:  # Evening
                    activity = random.choice(['socializing', 'idle', 'exercising'])
                else:  # Late evening
                    activity = random.choice(['idle', 'socializing'])
                
                hourly_activity.append(activity)
            
            pattern['hourly'] = hourly_activity
            
            # Focus level patterns
            focus_levels = []
            for hour in range(24):
                if hourly_activity[hour] == 'working':
                    focus_levels.append(random.uniform(0.7, 1.0))
                elif hourly_activity[hour] == 'sleeping':
                    focus_levels.append(0.0)
                elif hourly_activity[hour] == 'exercising':
                    focus_levels.append(random.uniform(0.3, 0.6))
                else:
                    focus_levels.append(random.uniform(0.2, 0.5))
            
            pattern['focus'] = focus_levels
            
            patterns[user_id] = pattern
        
        return patterns
    
    def _generate_notification_patterns(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Generate notification arrival patterns for each user"""
        patterns = {}
        
        for user_id, profile in self.user_profiles.items():
            pattern = {}
            
            for notif_type in self.notification_types:
                # Base arrival rates per hour
                if notif_type == 'social':
                    if profile['usage_pattern'] == 'social_media_heavy':
                        base_rate = 3.0
                    else:
                        base_rate = 1.0
                elif notif_type == 'messaging':
                    base_rate = 2.0
                elif notif_type == 'email':
                    if profile['usage_pattern'] == 'productive':
                        base_rate = 2.5
                    else:
                        base_rate = 1.0
                elif notif_type == 'system':
                    base_rate = 0.5
                elif notif_type == 'productivity':
                    if profile['usage_pattern'] == 'productive':
                        base_rate = 1.5
                    else:
                        base_rate = 0.5
                else:  # entertainment
                    base_rate = 1.0
                
                # Generate hourly rates with some variation
                hourly_rates = []
                for hour in range(24):
                    # Adjust rates based on time of day
                    if 9 <= hour <= 17:  # Work hours
                        multiplier = 1.2 if notif_type in ['email', 'productivity'] else 0.8
                    elif 18 <= hour <= 22:  # Evening
                        multiplier = 1.5 if notif_type in ['social', 'messaging', 'entertainment'] else 0.6
                    else:  # Night/early morning
                        multiplier = 0.2
                    
                    rate = base_rate * multiplier * random.uniform(0.8, 1.2)
                    hourly_rates.append(rate)
                
                pattern[notif_type] = np.array(hourly_rates)
            
            patterns[user_id] = pattern
        
        return patterns
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        # Select random user
        self.current_user_id = random.choice(list(self.user_profiles.keys()))
        
        # Reset time
        self.current_hour = 0
        self.current_day = 0
        
        # Reset user state
        activity_pattern = self.activity_patterns[self.current_user_id]
        self.user_activity = activity_pattern['hourly'][self.current_hour]
        self.focus_level = activity_pattern['focus'][self.current_hour]
        self.battery_level = 100.0
        
        # Reset notifications
        self.pending_notifications = []
        self.delivered_notifications = []
        self.notification_queue = []
        self.engagement_history = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Basic state
        state = np.array([
            self.current_hour / 24.0,
            self.current_day / 7.0,
            self._activity_to_numeric(self.user_activity),
            self.focus_level,
            self.battery_level / 100.0,
            len(self.pending_notifications) / 10.0  # Normalized pending count
        ])
        
        # Notification queue composition
        queue_composition = np.zeros(self.num_types)
        for notif in self.pending_notifications:
            idx = self.notification_types.index(notif['type'])
            queue_composition[idx] += 1
        
        # Normalize queue composition
        if len(self.pending_notifications) > 0:
            queue_composition = queue_composition / len(self.pending_notifications)
        
        state = np.concatenate([state, queue_composition])
        
        # Recent delivery performance (last 5 hours)
        recent_performance = np.zeros(5)
        if len(self.delivered_notifications) > 0:
            for i in range(min(5, len(self.delivered_notifications))):
                notif = self.delivered_notifications[-(i+1)]
                recent_performance[i] = notif.get('engagement', 0.0)
        
        state = np.concatenate([state, recent_performance])
        
        return state
    
    def _activity_to_numeric(self, activity: str) -> float:
        """Convert activity to numeric representation"""
        activity_map = {
            'sleeping': 0.0,
            'idle': 0.25,
            'exercising': 0.5,
            'socializing': 0.75,
            'working': 1.0
        }
        return activity_map.get(activity, 0.25)
    
    def _generate_notifications(self):
        """Generate new notifications for current hour"""
        pattern = self.notification_patterns[self.current_user_id]
        
        for notif_type in self.notification_types:
            # Poisson process for notification arrival
            rate = pattern[notif_type][self.current_hour]
            num_notifications = np.random.poisson(rate)
            
            for _ in range(num_notifications):
                notification = {
                    'type': notif_type,
                    'arrival_time': self.current_hour,
                    'urgency': random.uniform(0.1, 1.0),
                    'importance': random.uniform(0.1, 1.0),
                    'content_length': random.choice(['short', 'medium', 'long']),
                    'sender': f'{notif_type}_app',
                    'id': len(self.notification_queue)
                }
                
                self.pending_notifications.append(notification)
                self.notification_queue.append(notification)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time step"""
        # Generate new notifications for this hour
        self._generate_notifications()
        
        # Execute action on pending notifications
        delivered_this_step = []
        
        if action == 0:  # Deliver now
            # Deliver one most urgent notification
            if self.pending_notifications:
                most_urgent = max(self.pending_notifications, key=lambda x: x['urgency'] * x['importance'])
                self.pending_notifications.remove(most_urgent)
                delivered_this_step.append(most_urgent)
        
        elif action == 1:  # Delay 1 hour
            # Don't deliver anything, but may lose some urgent notifications
            lost = [n for n in self.pending_notifications if n['urgency'] > 0.8 and random.random() < 0.3]
            for notif in lost:
                self.pending_notifications.remove(notif)
        
        elif action == 2:  # Schedule optimal time
            # Find optimal notification based on context
            if self.pending_notifications:
                # Score notifications based on current context
                scored_notifications = []
                for notif in self.pending_notifications:
                    score = self._calculate_notification_score(notif)
                    scored_notifications.append((score, notif))
                
                # Deliver highest scored notification
                if scored_notifications:
                    scored_notifications.sort(key=lambda x: x[0], reverse=True)
                    best_notif = scored_notifications[0][1]
                    self.pending_notifications.remove(best_notif)
                    delivered_this_step.append(best_notif)
        
        # Calculate engagement for delivered notifications
        for notif in delivered_this_step:
            engagement = self._calculate_engagement(notif)
            notif['engagement'] = engagement
            notif['delivery_time'] = self.current_hour
            notif['context'] = self.user_activity
            self.delivered_notifications.append(notif)
            self.engagement_history.append(engagement)
        
        # Calculate reward
        reward = self._calculate_reward(delivered_this_step, action)
        
        # Update user state for next hour
        self._update_user_state()
        
        # Advance time
        self.current_hour += 1
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_day += 1
        
        # Check if episode is done
        done = (self.current_day >= 1) or (self.battery_level <= 10)
        
        # Prepare info dictionary
        info = {
            'user_id': self.current_user_id,
            'hour': self.current_hour,
            'activity': self.user_activity,
            'focus_level': self.focus_level,
            'pending_count': len(self.pending_notifications),
            'delivered_count': len(delivered_this_step),
            'total_delivered': len(self.delivered_notifications),
            'avg_engagement': np.mean(self.engagement_history) if self.engagement_history else 0.0
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_notification_score(self, notification: Dict[str, Any]) -> float:
        """Calculate score for notification based on current context"""
        score = 0.0
        
        # Base importance and urgency
        score += notification['importance'] * 0.4
        score += notification['urgency'] * 0.3
        
        # Contextual factors
        if self.user_activity == 'working':
            if notification['type'] in ['productivity', 'email']:
                score += 0.2
            elif notification['type'] in ['social', 'entertainment']:
                score -= 0.2
        elif self.user_activity == 'socializing':
            if notification['type'] in ['social', 'messaging']:
                score += 0.2
        elif self.user_activity == 'sleeping':
            # Only very urgent notifications during sleep
            if notification['urgency'] > 0.9:
                score += 0.1
            else:
                score -= 0.5
        
        # Focus level consideration
        if self.focus_level > 0.7:
            # High focus - prefer important notifications
            score += notification['importance'] * 0.2
        else:
            # Low focus - can deliver less important notifications
            score += 0.1
        
        return score
    
    def _calculate_engagement(self, notification: Dict[str, Any]) -> float:
        """Calculate user engagement with delivered notification"""
        engagement = 0.5  # Base engagement
        
        # Context affects engagement
        if self.user_activity == 'working':
            if notification['type'] in ['productivity', 'email']:
                engagement += 0.3
            elif notification['type'] in ['social', 'entertainment']:
                engagement -= 0.2
        elif self.user_activity == 'socializing':
            if notification['type'] in ['social', 'messaging']:
                engagement += 0.3
        elif self.user_activity == 'sleeping':
            engagement -= 0.4  # Poor engagement during sleep
        
        # Focus level affects engagement
        if self.focus_level > 0.7:
            engagement += 0.2  # Better engagement when focused
        
        # Urgency and importance
        engagement += notification['urgency'] * 0.1
        engagement += notification['importance'] * 0.1
        
        # Battery level affects engagement
        if self.battery_level < 20:
            engagement -= 0.2  # Less engagement when battery is low
        
        return np.clip(engagement, 0.0, 1.0)
    
    def _calculate_reward(self, delivered_notifications: List[Dict[str, Any]], action: int) -> float:
        """Calculate reward for notification scheduling decisions"""
        reward = 0.0
        
        # Reward for delivered notifications
        for notif in delivered_notifications:
            engagement = notif['engagement']
            reward += engagement * 2.0  # High weight for engagement
            
            # Bonus for timely delivery
            delay = self.current_hour - notif['arrival_time']
            if delay == 0:
                reward += 0.5  # Immediate delivery bonus
            elif delay <= 2:
                reward += 0.2  # Quick delivery bonus
        
        # Penalty for too many pending notifications
        pending_penalty = len(self.pending_notifications) * 0.1
        reward -= pending_penalty
        
        # Penalty for lost notifications (due to excessive delay)
        lost_notifications = [n for n in self.pending_notifications 
                              if (self.current_hour - n['arrival_time']) > 24]
        reward -= len(lost_notifications) * 0.5
        
        # Action-specific rewards
        if action == 0:  # Deliver now
            if self.focus_level < 0.3:  # Good timing for delivery
                reward += 0.2
            elif self.focus_level > 0.8:  # Bad timing for delivery
                reward -= 0.1
        
        elif action == 2:  # Schedule optimal time
            # Reward for intelligent scheduling
            if delivered_notifications:
                avg_engagement = np.mean([n['engagement'] for n in delivered_notifications])
                if avg_engagement > 0.7:
                    reward += 0.3
        
        # Battery consideration
        if self.battery_level > 50:
            reward += 0.05  # Small bonus for maintaining battery
        
        # Queue management bonus
        if len(self.pending_notifications) < 5:
            reward += 0.1
        
        return reward
    
    def _update_user_state(self):
        """Update user state for next hour"""
        # Update activity based on pattern
        activity_pattern = self.activity_patterns[self.current_user_id]
        self.user_activity = activity_pattern['hourly'][self.current_hour]
        self.focus_level = activity_pattern['focus'][self.current_hour]
        
        # Update battery level
        battery_drain = len(self.delivered_notifications) * 0.5  # 0.5% per notification
        self.battery_level -= battery_drain
        self.battery_level = max(0, self.battery_level)
    
    def get_notification_summary(self) -> Dict[str, Any]:
        """Get summary of notification performance"""
        if not self.delivered_notifications:
            return {
                'total_delivered': 0,
                'avg_engagement': 0.0,
                'pending_count': len(self.pending_notifications),
                'engagement_by_type': {}
            }
        
        engagement_by_type = {}
        for notif_type in self.notification_types:
            type_notifications = [n for n in self.delivered_notifications if n['type'] == notif_type]
            if type_notifications:
                engagement_by_type[notif_type] = np.mean([n['engagement'] for n in type_notifications])
            else:
                engagement_by_type[notif_type] = 0.0
        
        return {
            'total_delivered': len(self.delivered_notifications),
            'avg_engagement': np.mean([n['engagement'] for n in self.delivered_notifications]),
            'pending_count': len(self.pending_notifications),
            'engagement_by_type': engagement_by_type,
            'battery_level': self.battery_level
        }
