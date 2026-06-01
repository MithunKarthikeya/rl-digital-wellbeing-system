"""
Build Final Unified Dataset from 3 source datasets.

Sources:
  1. Mobile_Usage_Screentime_Dataset_.xlsx  (120 rows)
  2. Data.csv                               (3500 rows)
  3. Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv  (7500 rows)

Output: results/data/final_unified_dataset.csv  (11120 rows)

Unified columns (all numeric / encoded):
  user_id, data_source,
  age, gender_encoded,
  screen_time_hours,
  social_media_hours, gaming_hours, study_work_hours,
  sleep_hours,
  notifications_per_day,
  stress_level_norm,       # 0-1
  anxiety_score_norm,      # 0-1
  depression_score_norm,   # 0-1
  happiness_score_norm,    # 0-1
  focus_score_norm,        # 0-1
  productivity_score_norm, # 0-1
  digital_dependence_norm, # 0-1
  sleep_quality_norm,      # 0-1
  physical_activity_days,
  phone_unlocks,
  app_opens_per_day,
  weekend_screen_time,
  battery_drain_percent,
  addiction_severity,      # 0=None/No, 1=Mild, 2=Moderate, 3=Severe
  high_risk_flag,          # 0/1
  device_type_encoded,     # 0=Android,1=iPhone,2=Laptop,3=Tablet,4=Other
  region,                  # kept as string (only available from DATA.csv)
  income_level,            # kept as string (only available from DATA.csv)
  education_level,         # kept as string (only available from DATA.csv)
  daily_role               # kept as string (only available from DATA.csv)
"""

import os
import pandas as pd
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def encode_gender(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v in ('male', 'm'):
        return 1
    if v in ('female', 'f'):
        return 0
    return 0.5   # Other / non-binary

def encode_device(val):
    mapping = {'android': 0, 'iphone': 1, 'laptop': 2, 'tablet': 3}
    if pd.isna(val):
        return 4
    return mapping.get(str(val).strip().lower(), 4)

def encode_addiction(val):
    """Return 0-3 severity from label string or NaN."""
    if pd.isna(val):
        return 0
    mapping = {'no': 0, 'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    return mapping.get(str(val).strip().lower(), 0)

def norm(series, lo, hi):
    """Clip then min-max normalise to [0, 1]."""
    clipped = series.clip(lo, hi)
    return (clipped - lo) / (hi - lo)

# ── load sources ─────────────────────────────────────────────────────────────

print("Loading source datasets...")

mobile = pd.read_excel('Mobile_Usage_Screentime_Dataset_.xlsx')
lifestyle = pd.read_csv('Data.csv')
addiction = pd.read_csv('Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')

print(f"  Mobile Usage  : {mobile.shape}")
print(f"  Lifestyle     : {lifestyle.shape}")
print(f"  Addiction     : {addiction.shape}")

# ── 1. Mobile Usage ──────────────────────────────────────────────────────────

print("\nProcessing Mobile Usage dataset...")

m = pd.DataFrame()
m['user_id']                = mobile['UserID'].astype(int)
m['data_source']            = 'mobile_usage'
m['age']                    = mobile['Age']
m['gender_encoded']         = mobile['Gender'].apply(encode_gender)
m['screen_time_hours']      = mobile['Daily_ScreenTime_Hours']
m['social_media_hours']     = mobile['SocialMedia_Min'] / 60.0
m['gaming_hours']           = mobile['Gaming_Min'] / 60.0
m['study_work_hours']       = mobile['Study_Min'] / 60.0
m['sleep_hours']            = mobile['Sleep_Hours']
m['notifications_per_day']  = mobile['Messages_Sent']          # closest proxy
m['stress_level_norm']      = np.nan
m['anxiety_score_norm']     = np.nan
m['depression_score_norm']  = np.nan
m['happiness_score_norm']   = np.nan
m['focus_score_norm']       = np.nan
m['productivity_score_norm']= np.nan
m['digital_dependence_norm']= np.nan
m['sleep_quality_norm']     = np.nan
m['physical_activity_days'] = np.nan
m['phone_unlocks']          = np.nan
m['app_opens_per_day']      = np.nan
m['weekend_screen_time']    = np.nan
m['battery_drain_percent']  = mobile['Battery_Drain_Percent']
m['addiction_severity']     = 0
m['high_risk_flag']         = 0
m['device_type_encoded']    = mobile['Device Type'].apply(encode_device)
m['region']                 = np.nan
m['income_level']           = np.nan
m['education_level']        = np.nan
m['daily_role']             = np.nan

# ── 2. Lifestyle / DATA.csv ───────────────────────────────────────────────────

print("Processing Lifestyle dataset...")

l = pd.DataFrame()
l['user_id']                = lifestyle['id'].astype(int)
l['data_source']            = 'lifestyle'
l['age']                    = lifestyle['age']
l['gender_encoded']         = lifestyle['gender'].apply(encode_gender)
l['screen_time_hours']      = lifestyle['device_hours_per_day']
l['social_media_hours']     = lifestyle['social_media_mins'] / 60.0
l['gaming_hours']           = np.nan
l['study_work_hours']       = lifestyle['study_mins'] / 60.0
l['sleep_hours']            = lifestyle['sleep_hours']
l['notifications_per_day']  = lifestyle['notifications_per_day']
# Normalise scores to [0,1]
l['stress_level_norm']      = norm(lifestyle['stress_level'], 1, 10)
l['anxiety_score_norm']     = norm(lifestyle['anxiety_score'], 0, 27.2)
l['depression_score_norm']  = norm(lifestyle['depression_score'], 0, 27)
l['happiness_score_norm']   = norm(lifestyle['happiness_score'], 0, 10)
l['focus_score_norm']       = norm(lifestyle['focus_score'], 0, 100)
l['productivity_score_norm']= norm(lifestyle['productivity_score'], 33, 95)
l['digital_dependence_norm']= norm(lifestyle['digital_dependence_score'], 5.6, 89.2)
l['sleep_quality_norm']     = norm(lifestyle['sleep_quality'], 1, 5)
l['physical_activity_days'] = lifestyle['physical_activity_days']
l['phone_unlocks']          = lifestyle['phone_unlocks']
l['app_opens_per_day']      = np.nan
l['weekend_screen_time']    = np.nan
l['battery_drain_percent']  = np.nan
l['addiction_severity']     = lifestyle['high_risk_flag'].astype(int)  # 0/1 risk flag
l['high_risk_flag']         = lifestyle['high_risk_flag'].astype(int)
l['device_type_encoded']    = lifestyle['device_type'].apply(encode_device)
l['region']                 = lifestyle['region']
l['income_level']           = lifestyle['income_level']
l['education_level']        = lifestyle['education_level']
l['daily_role']             = lifestyle['daily_role']

# ── 3. Addiction Analysis ─────────────────────────────────────────────────────

print("Processing Addiction Analysis dataset...")

# Map categorical stress to numeric
stress_map = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8}

a = pd.DataFrame()
a['user_id']                = range(1, len(addiction) + 1)   # generate sequential IDs
a['data_source']            = 'addiction_analysis'
a['age']                    = addiction['age']
a['gender_encoded']         = addiction['gender'].apply(encode_gender)
a['screen_time_hours']      = addiction['daily_screen_time_hours']
a['social_media_hours']     = addiction['social_media_hours']
a['gaming_hours']           = addiction['gaming_hours']
a['study_work_hours']       = addiction['work_study_hours']
a['sleep_hours']            = addiction['sleep_hours']
a['notifications_per_day']  = addiction['notifications_per_day']
a['stress_level_norm']      = addiction['stress_level'].map(stress_map)
a['anxiety_score_norm']     = np.nan
a['depression_score_norm']  = np.nan
a['happiness_score_norm']   = np.nan
a['focus_score_norm']       = np.nan
a['productivity_score_norm']= np.nan
a['digital_dependence_norm']= np.nan
a['sleep_quality_norm']     = np.nan
a['physical_activity_days'] = np.nan
a['phone_unlocks']          = np.nan
a['app_opens_per_day']      = addiction['app_opens_per_day']
a['weekend_screen_time']    = addiction['weekend_screen_time']
a['battery_drain_percent']  = np.nan
a['addiction_severity']     = addiction['addiction_level'].apply(encode_addiction)
# high_risk_flag: addicted_label==1 OR addiction_level in Moderate/Severe
a['high_risk_flag']         = (
    (addiction['addicted_label'] == 1) |
    (addiction['addiction_level'].isin(['Moderate', 'Severe']))
).astype(int)
a['device_type_encoded']    = 4   # not available in this dataset
a['region']                 = np.nan
a['income_level']           = np.nan
a['education_level']        = np.nan
a['daily_role']             = np.nan

# ── Combine ───────────────────────────────────────────────────────────────────

print("\nCombining all datasets...")

unified = pd.concat([m, l, a], ignore_index=True)

# Ensure correct dtypes
int_cols = ['age', 'addiction_severity', 'high_risk_flag', 'device_type_encoded']
for col in int_cols:
    unified[col] = pd.to_numeric(unified[col], errors='coerce')

# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs('results/data', exist_ok=True)
out_path = 'results/data/final_unified_dataset.csv'
unified.to_csv(out_path, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Final Unified Dataset saved → {out_path}")
print(f"{'='*55}")
print(f"  Total rows    : {len(unified):,}")
print(f"  Total columns : {len(unified.columns)}")
print(f"\n  Rows per source:")
for src, cnt in unified['data_source'].value_counts().items():
    print(f"    {src:<25} {cnt:>5} rows")

print(f"\n  Columns:")
for col in unified.columns:
    non_null = unified[col].notna().sum()
    pct = non_null / len(unified) * 100
    print(f"    {col:<30} {non_null:>6} non-null  ({pct:5.1f}%)")

print(f"\n  Numeric summary:")
print(unified.describe().round(3).to_string())
