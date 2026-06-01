import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedDataProcessor:
    """Enhanced data processor for multiple datasets with different schemas"""
    
    def __init__(self, datasets: Dict[str, str]):
        self.datasets = datasets
        self.data = {}
        self.scalers = {}
        self.encoders = {}
        self.combined_profiles = {}
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets"""
        print("📊 Loading multiple datasets...")
        
        for name, path in self.datasets.items():
            try:
                if path.endswith('.xlsx'):
                    self.data[name] = pd.read_excel(path)
                elif path.endswith('.csv'):
                    self.data[name] = pd.read_csv(path)
                else:
                    print(f"⚠️  Unsupported file format for {name}: {path}")
                    continue
                    
                print(f"✅ Loaded {name}: {self.data[name].shape}")
                
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
                continue
                
        return self.data
    
    def explore_all_datasets(self) -> Dict[str, Any]:
        """Perform exploratory data analysis on all datasets"""
        print("\n🔍 Exploring All Datasets...")
        
        exploration = {}
        for name, df in self.data.items():
            exploration[name] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'missing_values': df.isnull().sum().sum(),
                'dtypes': df.dtypes.to_dict()
            }
            
            print(f"\n📈 {name}:")
            print(f"   Shape: {exploration[name]['shape']}")
            print(f"   Columns: {len(exploration[name]['columns'])}")
            print(f"   Missing values: {exploration[name]['missing_values']}")
            
        return exploration
    
    def create_unified_user_profiles(self) -> Dict[int, Dict[str, Any]]:
        """Create unified user profiles combining all datasets"""
        print("\n👥 Creating unified user profiles...")
        
        # Start with the most comprehensive dataset (DATA.csv)
        if 'DATA' in self.data:
            base_profiles = self._create_profiles_from_lifestyle_data()
        else:
            base_profiles = {}
        
        # Enhance with mobile usage data
        if 'Mobile_Usage' in self.data:
            usage_profiles = self._create_profiles_from_usage_data()
            base_profiles = self._merge_profiles(base_profiles, usage_profiles)
        
        # Enhance with addiction analysis data
        if 'Addiction' in self.data:
            addiction_profiles = self._create_profiles_from_addiction_data()
            base_profiles = self._merge_profiles(base_profiles, addiction_profiles)
        
        self.combined_profiles = base_profiles
        print(f"✅ Created {len(self.combined_profiles)} unified user profiles")
        
        return self.combined_profiles
    
    def _create_profiles_from_lifestyle_data(self) -> Dict[int, Dict[str, Any]]:
        """Create profiles from DATA.csv (lifestyle dataset)"""
        df = self.data['DATA'].copy()
        
        profiles = {}
        for _, row in df.iterrows():
            user_id = int(row['id'])
            
            # Calculate screen time category
            daily_hours = float(row['device_hours_per_day'])
            if daily_hours < 4:
                screen_category = 'low'
            elif daily_hours < 8:
                screen_category = 'moderate'
            else:
                screen_category = 'high'
            
            # Determine usage pattern
            social_media_ratio = float(row['social_media_mins']) / max(daily_hours * 60, 1)
            study_ratio = float(row['study_mins']) / max(daily_hours * 60, 1)
            
            if social_media_ratio > 0.4:
                usage_pattern = 'social_media_heavy'
            elif study_ratio > 0.5:
                usage_pattern = 'productive'
            else:
                usage_pattern = 'balanced'
            
            # Sleep quality
            sleep_quality = 'good' if row['sleep_quality'] >= 7 else 'adequate' if row['sleep_quality'] >= 5 else 'poor'
            
            profiles[user_id] = {
                'age': int(row['age']),
                'gender': row['gender'],
                'region': row['region'],
                'income_level': row['income_level'],
                'education_level': row['education_level'],
                'daily_role': row['daily_role'],
                'device_hours_per_day': daily_hours,
                'phone_unlocks': int(row['phone_unlocks']),
                'notifications_per_day': int(row['notifications_per_day']),
                'social_media_mins': float(row['social_media_mins']),
                'study_mins': float(row['study_mins']),
                'physical_activity_days': int(row['physical_activity_days']),
                'sleep_hours': float(row['sleep_hours']),
                'sleep_quality': sleep_quality,
                'anxiety_score': float(row['anxiety_score']),
                'depression_score': float(row['depression_score']),
                'stress_level': float(row['stress_level']),
                'happiness_score': float(row['happiness_score']),
                'focus_score': float(row['focus_score']),
                'high_risk_flag': bool(row['high_risk_flag']),
                'device_type': row['device_type'],
                'productivity_score': float(row['productivity_score']),
                'digital_dependence_score': float(row['digital_dependence_score']),
                'screen_time_category': screen_category,
                'usage_pattern': usage_pattern,
                'data_source': 'lifestyle'
            }
            
        return profiles
    
    def _create_profiles_from_usage_data(self) -> Dict[int, Dict[str, Any]]:
        """Create profiles from mobile usage dataset"""
        df = self.data['Mobile_Usage'].copy()
        
        profiles = {}
        for _, row in df.iterrows():
            user_id = int(row['UserID'])
            
            profiles[user_id] = {
                'age': int(row['Age']),
                'gender': row['Gender'],
                'daily_screen_time_hours': float(row['Daily_ScreenTime_Hours']),
                'social_media_min': float(row['SocialMedia_Min']),
                'gaming_min': float(row['Gaming_Min']),
                'study_min': float(row['Study_Min']),
                'sleep_hours': float(row['Sleep_Hours']),
                'device_type': row['Device Type'],
                'calls_min': float(row['Calls_Min']),
                'messages_sent': int(row['Messages_Sent']),
                'battery_drain_percent': float(row['Battery_Drain_Percent']),
                'data_source': 'mobile_usage'
            }
            
        return profiles
    
    def _create_profiles_from_addiction_data(self) -> Dict[int, Dict[str, Any]]:
        """Create profiles from addiction analysis dataset"""
        df = self.data['Addiction'].copy()
        
        profiles = {}
        for _, row in df.iterrows():
            user_id = int(row['user_id'])
            
            # Determine addiction level
            addiction_level = row['addicted_label']
            if addiction_level == 'No':
                addiction_severity = 0
            elif addiction_level == 'Mild':
                addiction_severity = 1
            elif addiction_level == 'Moderate':
                addiction_severity = 2
            else:  # Severe
                addiction_severity = 3
            
            profiles[user_id] = {
                'age': int(row['age']),
                'gender': row['gender'],
                'daily_screen_time_hours': float(row['daily_screen_time_hours']),
                'social_media_hours': float(row['social_media_hours']),
                'gaming_hours': float(row['gaming_hours']),
                'work_study_hours': float(row['work_study_hours']),
                'sleep_hours': float(row['sleep_hours']),
                'notifications_per_day': int(row['notifications_per_day']),
                'app_opens_per_day': int(row['app_opens_per_day']),
                'weekend_screen_time': float(row['weekend_screen_time']),
                'stress_level': float(row['stress_level']),
                'academic_work_impact': float(row['academic_work_impact']),
                'addiction_level': addiction_level,
                'addiction_severity': addiction_severity,
                'data_source': 'addiction_analysis'
            }
            
        return profiles
    
    def _merge_profiles(self, base_profiles: Dict, enhancement_profiles: Dict) -> Dict:
        """Merge profile dictionaries intelligently"""
        merged = base_profiles.copy()
        
        for user_id, enhancement in enhancement_profiles.items():
            if user_id in merged:
                # Merge with existing profile
                for key, value in enhancement.items():
                    if key not in merged[user_id]:
                        merged[user_id][key] = value
                    elif key in ['age', 'gender'] and merged[user_id][key] is None:
                        merged[user_id][key] = value
                    elif key in ['daily_screen_time_hours', 'stress_level', 'sleep_hours']:
                        # Average the values if both exist
                        if merged[user_id][key] is not None:
                            merged[user_id][f'{key}_avg'] = (merged[user_id][key] + value) / 2
                        merged[user_id][key] = value
            else:
                # Add new user
                merged[user_id] = enhancement
        
        # Handle user ID conflicts by creating new IDs for overlapping users
        max_user_id = max(list(merged.keys()) + list(enhancement_profiles.keys()), default=0)
        
        # Create new unified profiles with consistent IDs
        unified_profiles = {}
        current_id = 1
        
        # First, add all base profiles
        for user_id, profile in base_profiles.items():
            unified_profiles[current_id] = profile
            current_id += 1
        
        # Then, add all enhancement profiles with new IDs
        for user_id, profile in enhancement_profiles.items():
            unified_profiles[current_id] = profile
            current_id += 1
        
        # Handle empty profiles case
        if not unified_profiles:
            print("⚠️  Warning: No user profiles created from datasets")
            # Create dummy profile for testing
            for i in range(10):  # Create 10 dummy profiles for testing
                unified_profiles[i+1] = {
                    'age': 25 + i,
                    'gender': 'Male' if i % 2 == 0 else 'Female',
                    'data_source': 'dummy'
                }
        
        return unified_profiles
    
    def get_enhanced_features(self, user_id: int) -> np.ndarray:
        """Get enhanced feature vector for RL training"""
        if user_id not in self.combined_profiles:
            # Return default features if user not found
            return np.zeros(20)  # 20-dimensional feature space
        
        profile = self.combined_profiles[user_id]
        
        features = []
        
        # Basic demographics
        features.append(profile.get('age', 25) / 50.0)  # Normalized age
        features.append(1.0 if profile.get('gender', 'Male') == 'Male' else 0.0)
        
        # Screen time and usage patterns
        daily_screen_time = profile.get('daily_screen_time_hours', 6.0)
        features.append(min(daily_screen_time / 12.0, 1.0))  # Normalized to 12 hours
        
        # Social media usage
        social_media_hours = profile.get('social_media_hours', profile.get('social_media_mins', 0) / 60.0)
        features.append(min(social_media_hours / 8.0, 1.0))  # Normalized to 8 hours
        
        # Gaming usage
        gaming_hours = profile.get('gaming_hours', profile.get('gaming_min', 0) / 60.0)
        features.append(min(gaming_hours / 6.0, 1.0))  # Normalized to 6 hours
        
        # Productivity
        work_study_hours = profile.get('work_study_hours', profile.get('study_mins', 0) / 60.0)
        features.append(min(work_study_hours / 10.0, 1.0))  # Normalized to 10 hours
        
        # Sleep metrics
        sleep_hours = profile.get('sleep_hours', 7.0)
        features.append(min(sleep_hours / 10.0, 1.0))  # Normalized to 10 hours
        
        # Mental health scores
        features.append(profile.get('stress_level', 5.0) / 10.0)  # Normalized stress
        features.append(profile.get('happiness_score', 5.0) / 10.0)  # Normalized happiness
        features.append(profile.get('focus_score', 5.0) / 10.0)  # Normalized focus
        features.append(profile.get('anxiety_score', 5.0) / 10.0)  # Normalized anxiety
        features.append(profile.get('depression_score', 5.0) / 10.0)  # Normalized depression
        
        # Behavioral metrics
        features.append(profile.get('notifications_per_day', 50) / 100.0)  # Normalized notifications
        features.append(profile.get('phone_unlocks', 100) / 200.0)  # Normalized unlocks
        features.append(profile.get('productivity_score', 50) / 100.0)  # Normalized productivity
        features.append(profile.get('digital_dependence_score', 50) / 100.0)  # Normalized digital dependence
        
        # Addiction severity (if available)
        features.append(profile.get('addiction_severity', 0) / 3.0)  # Normalized addiction severity
        
        # Device type encoding
        device_type = profile.get('device_type', 'Android')
        features.append(1.0 if device_type == 'iPhone' else 0.0)
        
        # Risk flag
        features.append(1.0 if profile.get('high_risk_flag', False) else 0.0)
        
        # Physical activity
        physical_days = profile.get('physical_activity_days', 3)
        features.append(physical_days / 7.0)  # Normalized to 7 days
        
        # Academic/work impact
        academic_impact = profile.get('academic_work_impact', 5.0)
        features.append(min(academic_impact / 10.0, 1.0))  # Normalized impact
        
        return np.array(features)
    
    def visualize_dataset_comparison(self):
        """Create comprehensive visualization comparing all datasets"""
        print("\n📈 Creating dataset comparison visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Dataset 1: DATA.csv - Mental Health & Wellbeing
        if 'DATA' in self.data:
            df = self.data['DATA']
            
            # Mental health distribution
            mental_cols = ['anxiety_score', 'depression_score', 'stress_level', 'happiness_score']
            for i, col in enumerate(mental_cols):
                axes[0, 0].hist(df[col], bins=20, alpha=0.7, label=col.replace('_', ' ').title())
            axes[0, 0].set_title('Mental Health Scores Distribution')
            axes[0, 0].legend()
            
            # Screen time by education level
            education_screen = df.groupby('education_level')['device_hours_per_day'].mean()
            axes[0, 1].bar(education_screen.index, education_screen.values)
            axes[0, 1].set_title('Screen Time by Education Level')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Digital dependence vs productivity
            axes[0, 2].scatter(df['digital_dependence_score'], df['productivity_score'], alpha=0.6)
            axes[0, 2].set_xlabel('Digital Dependence Score')
            axes[0, 2].set_ylabel('Productivity Score')
            axes[0, 2].set_title('Digital Dependence vs Productivity')
        
        # Dataset 2: Mobile Usage - App Usage Patterns
        if 'Mobile_Usage' in self.data:
            df = self.data['Mobile_Usage']
            
            # Age vs screen time
            axes[1, 0].scatter(df['Age'], df['Daily_ScreenTime_Hours'], alpha=0.6)
            axes[1, 0].set_xlabel('Age')
            axes[1, 0].set_ylabel('Daily Screen Time (Hours)')
            axes[1, 0].set_title('Age vs Screen Time')
            
            # Usage breakdown
            usage_cols = ['SocialMedia_Min', 'Gaming_Min', 'Study_Min', 'Calls_Min']
            usage_means = [df[col].mean() for col in usage_cols]
            axes[1, 1].bar(usage_cols, usage_means)
            axes[1, 1].set_title('Average Usage by Category (Minutes)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Device type distribution
            device_counts = df['Device Type'].value_counts()
            axes[1, 2].pie(device_counts.values, labels=device_counts.index, autopct='%1.1f%%')
            axes[1, 2].set_title('Device Type Distribution')
        
        # Dataset 3: Addiction Analysis - Risk Patterns
        if 'Addiction' in self.data:
            df = self.data['Addiction']
            
            # Addiction levels
            addiction_counts = df['addicted_label'].value_counts()
            axes[1, 3].bar(addiction_counts.index, addiction_counts.values)
            axes[1, 3].set_title('Addiction Level Distribution')
            axes[1, 3].tick_params(axis='x', rotation=45)
            
            # Screen time vs addiction
            addiction_mapping = {'No': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
            df['addiction_severity'] = df['addicted_label'].map(addiction_mapping)
            axes[1, 4].scatter(df['daily_screen_time_hours'], df['addiction_severity'], alpha=0.6)
            axes[1, 4].set_xlabel('Daily Screen Time (Hours)')
            axes[1, 4].set_ylabel('Addiction Severity')
            axes[1, 4].set_title('Screen Time vs Addiction Severity')
        
        plt.tight_layout()
        plt.savefig('results/data/dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_data_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        summary = {
            'total_users': len(self.combined_profiles),
            'datasets_used': list(self.data.keys()),
            'feature_coverage': {},
            'data_quality': {}
        }
        
        # Feature coverage analysis
        all_features = set()
        for profile in self.combined_profiles.values():
            all_features.update(profile.keys())
        
        for feature in all_features:
            count = sum(1 for p in self.combined_profiles.values() if feature in p and p[feature] is not None)
            summary['feature_coverage'][feature] = {
                'count': count,
                'coverage': count / len(self.combined_profiles)
            }
        
        # Data quality metrics
        for name, df in self.data.items():
            summary['data_quality'][name] = {
                'completeness': (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                'size': df.shape[0],
                'features': df.shape[1]
            }
        
        return summary
