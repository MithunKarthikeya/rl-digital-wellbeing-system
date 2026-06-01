import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    """Data preprocessing and exploration utilities for Digital Wellbeing RL System"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the mobile usage dataset"""
        self.data = pd.read_excel(self.data_path)
        return self.data
    
    def explore_data(self) -> Dict[str, Any]:
        """Perform exploratory data analysis"""
        if self.data is None:
            self.load_data()
            
        exploration = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'statistics': self.data.describe().to_dict()
        }
        
        return exploration
    
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for RL training"""
        if self.data is None:
            self.load_data()
            
        # Create copies for processing
        processed_data = self.data.copy()
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Device Type']
        for col in categorical_cols:
            if col in processed_data.columns:
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col])
                self.encoders[col] = le
        
        # Create features for RL environments
        features = processed_data[['Age', 'Gender', 'Device Type', 
                                  'Daily_ScreenTime_Hours', 'SocialMedia_Min', 
                                  'Gaming_Min', 'Study_Min', 'Sleep_Hours',
                                  'Calls_Min', 'Messages_Sent', 'Battery_Drain_Percent']]
        
        # Normalize numerical features
        numerical_cols = ['Age', 'Daily_ScreenTime_Hours', 'SocialMedia_Min', 
                         'Gaming_Min', 'Study_Min', 'Sleep_Hours', 'Calls_Min', 
                         'Messages_Sent', 'Battery_Drain_Percent']
        
        features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
        
        return features.values, processed_data['UserID'].values
    
    def create_user_profiles(self) -> Dict[int, Dict[str, Any]]:
        """Create user profiles for personalization"""
        if self.data is None:
            self.load_data()
            
        profiles = {}
        for _, row in self.data.iterrows():
            user_id = row['UserID']
            profiles[user_id] = {
                'age': row['Age'],
                'gender': row['Gender'],
                'device_type': row['Device Type'],
                'screen_time_category': self._categorize_screen_time(row['Daily_ScreenTime_Hours']),
                'sleep_quality': self._categorize_sleep(row['Sleep_Hours']),
                'usage_pattern': self._analyze_usage_pattern(row)
            }
        
        return profiles
    
    def _categorize_screen_time(self, hours: float) -> str:
        """Categorize screen time usage"""
        if hours < 3:
            return 'low'
        elif hours < 7:
            return 'moderate'
        else:
            return 'high'
    
    def _categorize_sleep(self, hours: float) -> str:
        """Categorize sleep quality"""
        if hours < 6:
            return 'poor'
        elif hours < 8:
            return 'adequate'
        else:
            return 'good'
    
    def _analyze_usage_pattern(self, user_row) -> str:
        """Analyze user's digital usage pattern"""
        social_media_ratio = user_row['SocialMedia_Min'] / (user_row['Daily_ScreenTime_Hours'] * 60)
        gaming_ratio = user_row['Gaming_Min'] / (user_row['Daily_ScreenTime_Hours'] * 60)
        study_ratio = user_row['Study_Min'] / (user_row['Daily_ScreenTime_Hours'] * 60)
        
        if study_ratio > 0.5:
            return 'productive'
        elif social_media_ratio > 0.4:
            return 'social_media_heavy'
        elif gaming_ratio > 0.3:
            return 'gaming_heavy'
        else:
            return 'balanced'
    
    def visualize_data(self, save_path: str = None):
        """Create visualizations for data exploration"""
        if self.data is None:
            self.load_data()
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Screen time distribution
        axes[0, 0].hist(self.data['Daily_ScreenTime_Hours'], bins=20, alpha=0.7)
        axes[0, 0].set_title('Daily Screen Time Distribution')
        axes[0, 0].set_xlabel('Hours')
        
        # Age vs Screen Time
        axes[0, 1].scatter(self.data['Age'], self.data['Daily_ScreenTime_Hours'], alpha=0.6)
        axes[0, 1].set_title('Age vs Screen Time')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Screen Time (Hours)')
        
        # Usage breakdown
        usage_cols = ['SocialMedia_Min', 'Gaming_Min', 'Study_Min', 'Calls_Min']
        usage_means = [self.data[col].mean() for col in usage_cols]
        axes[0, 2].bar(usage_cols, usage_means)
        axes[0, 2].set_title('Average Usage by Category (Minutes)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Sleep vs Screen Time
        axes[1, 0].scatter(self.data['Sleep_Hours'], self.data['Daily_ScreenTime_Hours'], alpha=0.6)
        axes[1, 0].set_title('Sleep vs Screen Time')
        axes[1, 0].set_xlabel('Sleep Hours')
        axes[1, 0].set_ylabel('Screen Time (Hours)')
        
        # Device type distribution
        device_counts = self.data['Device Type'].value_counts()
        axes[1, 1].pie(device_counts.values, labels=device_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Device Type Distribution')
        
        # Gender distribution
        gender_counts = self.data['Gender'].value_counts()
        axes[1, 2].bar(gender_counts.index, gender_counts.values)
        axes[1, 2].set_title('Gender Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_state_representation(self, user_id: int) -> np.ndarray:
        """Get state representation for a specific user"""
        if self.data is None:
            self.load_data()
            
        user_data = self.data[self.data['UserID'] == user_id].iloc[0]
        
        # Create state vector
        state = np.array([
            user_data['Age'],
            1 if user_data['Gender'] == 'Male' else 0,
            1 if user_data['Device Type'] == 'iPhone' else 0,
            user_data['Daily_ScreenTime_Hours'],
            user_data['SocialMedia_Min'] / 60,  # Convert to hours
            user_data['Gaming_Min'] / 60,
            user_data['Study_Min'] / 60,
            user_data['Sleep_Hours'],
            user_data['Calls_Min'] / 60,
            user_data['Messages_Sent'] / 60,
            user_data['Battery_Drain_Percent']
        ])
        
        return state
