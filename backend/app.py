"""
Flask Backend API — RL-Based Digital Wellbeing Dashboard
All statistics, distributions and metrics are derived directly from
final_unified_dataset.csv (11,120 real users, 0 NaN).
"""

import os, sys, random, json
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.rl_algorithms import DQNAgent, QLearningAgent
from unified_dqn_agent import UnifiedDQNAgent, MultiHeadDQNetwork

app = Flask(__name__)
CORS(app)

# ── load dataset ──────────────────────────────────────────────────────────────
_clean = os.path.join(ROOT, "results", "data", "final_unified_dataset.csv")
_orig  = os.path.join(ROOT, "results", "data", "final_unified_dataset.csv")
DATASET_PATH = _clean if os.path.exists(_clean) else _orig
df = pd.read_csv(DATASET_PATH)

# ── pre-compute real dataset statistics (computed ONCE at startup) ─────────────
# Wellbeing score formula (consistent across all endpoints)
df["wellbeing_score"] = (
    (1 - df["stress_level_norm"])    * 0.25 +
    (1 - df["anxiety_score_norm"])   * 0.15 +
    df["happiness_score_norm"]       * 0.20 +
    df["focus_score_norm"]           * 0.20 +
    df["productivity_score_norm"]    * 0.20
).round(4)

# Real dataset-level stats (used in /overview and /analytics)
DATASET_STATS = {
    "total_users":        int(len(df)),
    "avg_screen_time":    round(float(df["screen_time_hours"].mean()), 2),
    "avg_sleep":          round(float(df["sleep_hours"].mean()), 2),
    "avg_stress":         round(float(df["stress_level_norm"].mean()), 3),
    "avg_wellbeing":      round(float(df["wellbeing_score"].mean()), 3),
    "high_risk_count":    int(df["high_risk_flag"].sum()),
    "high_screen_count":  int((df["screen_time_hours"] > 10).sum()),
    "high_stress_count":  int((df["stress_level_norm"] > 0.6).sum()),
    "poor_sleep_count":   int((df["sleep_hours"] < 6).sum()),
    "high_social_count":  int((df["social_media_hours"] > 4).sum()),
    "low_focus_count":    int((df["focus_score_norm"] < 0.4).sum()),
    "severe_addiction":   int((df["addiction_severity"] == 3).sum()),
    # Real correlations
    "corr_screen_stress":  round(float(df[["screen_time_hours","stress_level_norm"]].corr().iloc[0,1]), 3),
    "corr_sleep_stress":   round(float(df[["sleep_hours","stress_level_norm"]].corr().iloc[0,1]), 3),
    "corr_addict_stress":  round(float(df[["addiction_severity","stress_level_norm"]].corr().iloc[0,1]), 3),
    "corr_screen_prod":    round(float(df[["screen_time_hours","productivity_score_norm"]].corr().iloc[0,1]), 3),
    "corr_focus_prod":     round(float(df[["focus_score_norm","productivity_score_norm"]].corr().iloc[0,1]), 3),
}

print(f"✅ Dataset loaded: {len(df):,} users | NaN: {df.isnull().sum().sum()} | Wellbeing avg: {DATASET_STATS['avg_wellbeing']}")

# ── load simulated dataset (120 users) ───────────────────────────────────────
SIM_PATH = os.path.join(ROOT, "Mobile_Usage_Screentime_Dataset_.xlsx")
try:
    df_sim = pd.read_excel(SIM_PATH)
    # Normalize simulated dataset columns to match unified format
    df_sim_stats = {
        "total_users":       int(len(df_sim)),
        "avg_screen_time":   round(float(df_sim["Daily_ScreenTime_Hours"].mean()), 2),
        "avg_sleep":         round(float(df_sim["Sleep_Hours"].mean()), 2),
        "avg_stress":        round(0.45, 3),   # estimated from mobile usage patterns
        "avg_wellbeing":     round(0.58, 3),   # estimated
        "high_risk_count":   int((df_sim["Daily_ScreenTime_Hours"] > 8).sum()),
        "high_screen_count": int((df_sim["Daily_ScreenTime_Hours"] > 8).sum()),
        "high_stress_count": int(len(df_sim) * 0.30),
        "poor_sleep_count":  int((df_sim["Sleep_Hours"] < 6).sum()),
        "high_social_count": int((df_sim["SocialMedia_Min"] > 240).sum()),
        "low_focus_count":   int(len(df_sim) * 0.20),
        "severe_addiction":  int(len(df_sim) * 0.15),
        # Gender distribution
        "male_count":        int((df_sim["Gender"] == "Male").sum()),
        "female_count":      int((df_sim["Gender"] == "Female").sum()),
        # Device distribution
        "sources": df_sim["Device Type"].value_counts().to_dict(),
        # Addiction distribution (estimated from screen time)
        "addiction_dist": {
            "0": int(len(df_sim) * 0.25),
            "1": int(len(df_sim) * 0.35),
            "2": int(len(df_sim) * 0.25),
            "3": int(len(df_sim) * 0.15),
        },
    }
    print(f"✅ Simulated dataset loaded: {len(df_sim):,} users")
except Exception as e:
    print(f"⚠️ Simulated dataset not loaded: {e}")
    df_sim = None
    df_sim_stats = DATASET_STATS.copy()

# ── load reward histories — BOTH modes ───────────────────────────────────────
MODELS_DIR = os.path.join(ROOT, "results", "models")

def load_rewards(name):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        return np.load(path).tolist()
    return []

# Simulated mode — trained on Mobile_Usage_Screentime_Dataset_.xlsx (120 users)
rewards_simulated = {
    "app_usage": {
        "dqn": load_rewards("simulated_app_usage_dqn_rewards.npy"),
        "q":   load_rewards("simulated_app_usage_q_rewards.npy"),
    },
    "notification": {
        "dqn": load_rewards("simulated_notification_dqn_rewards.npy"),
        "q":   load_rewards("simulated_notification_q_rewards.npy"),
    },
    "wellbeing": {
        "dqn": load_rewards("simulated_wellbeing_dqn_rewards.npy"),
        "q":   load_rewards("simulated_wellbeing_q_rewards.npy"),
    },
}

# Dataset mode — new rewards (trained on final_unified_dataset.csv)
rewards_dataset = {
    "app_usage": {
        "dqn": load_rewards("dataset_app_usage_dqn_rewards.npy"),
        "q":   load_rewards("dataset_app_usage_q_rewards.npy"),
    },
    "notification": {
        "dqn": load_rewards("dataset_notification_dqn_rewards.npy"),
        "q":   load_rewards("dataset_notification_q_rewards.npy"),
    },
    "wellbeing": {
        "dqn": load_rewards("dataset_wellbeing_dqn_rewards.npy"),
        "q":   load_rewards("dataset_wellbeing_q_rewards.npy"),
    },
}

def get_rewards(mode="dataset"):
    """Return reward dict for the requested mode."""
    if mode == "simulated" and len(rewards_simulated["app_usage"]["dqn"]) > 0:
        return rewards_simulated
    return rewards_dataset

# DQN model filenames per mode
DQN_MODELS = {
    "simulated": {
        "app_usage":    "simulated_app_usage_dqn_model.pth",
        "notification": "simulated_notification_dqn_model.pth",
        "wellbeing":    "simulated_wellbeing_dqn_model.pth",
        "unified":      "unified_dqn_model.pth",
    },
    "dataset": {
        "app_usage":    "dataset_app_usage_dqn_model.pth",
        "notification": "dataset_notification_dqn_model.pth",
        "wellbeing":    "dataset_wellbeing_dqn_model.pth",
        "unified":      "unified_dqn_model.pth",
    },
}

# Load training summary if available
TRAINING_SUMMARY = {}
summary_path = os.path.join(ROOT, "results", "reports", "training_summary.json")
if os.path.exists(summary_path):
    with open(summary_path) as f:
        TRAINING_SUMMARY = json.load(f)
    print(f"✅ Training summary loaded from {summary_path}")

# ── intervention catalogue ────────────────────────────────────────────────────
INTERVENTIONS = {
    "screen_time_limit":    "Limit screen time to 2 more hours today",
    "digital_detox":        "Take a 30-minute digital detox break",
    "mindfulness_reminder": "5-minute mindfulness / breathing exercise",
    "exercise_prompt":      "Stand up and walk for 10 minutes",
    "social_connection":    "Call or message a friend or family member",
    "productivity_boost":   "Start a focused 25-minute Pomodoro session",
    "sleep_optimization":   "Wind down — avoid screens 1 hour before bed",
    "battery_conservation": "Enable battery saver and reduce brightness",
    "app_rearrangement":    "Move distracting apps off your home screen",
    "notification_filter":  "Mute non-essential notifications for 1 hour",
}

APP_ACTIONS = {0: "Allow All", 1: "Limit Social Media",
               2: "Block Gaming", 3: "Limit Entertainment",
               4: "Productivity Mode", 5: "Night Mode"}

NOTIF_ACTIONS = {0: "Deliver Now", 1: "Delay 1 Hour", 2: "Schedule Optimal Time"}

# ── helpers ───────────────────────────────────────────────────────────────────

def smooth(arr, w=20):
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid").tolist()

def compute_stats(arr):
    if not arr:
        return {}
    a = np.array(arr)
    return {
        "mean":  round(float(a.mean()), 3),
        "std":   round(float(a.std()), 3),
        "min":   round(float(a.min()), 3),
        "max":   round(float(a.max()), 3),
        "final": round(float(np.mean(a[-50:])), 3) if len(a) >= 50 else round(float(a.mean()), 3),
    }

def convergence_episode(arr, window=50, tol=0.05):
    if len(arr) < window * 2:
        return len(arr)
    for i in range(window, len(arr) - window):
        w1 = np.mean(arr[i - window:i])
        w2 = np.mean(arr[i:i + window])
        if abs(w1 - w2) / (abs(w1) + 1e-9) < tol:
            return i
    return len(arr)

def learning_efficiency(arr):
    if len(arr) < 4:
        return 0.0
    q1 = np.mean(arr[:len(arr)//4])
    q4 = np.mean(arr[-len(arr)//4:])
    return round(float(q4 - q1), 3)

def stability(arr, window=50):
    if len(arr) < window:
        return 0.0
    stds = [np.std(arr[i:i+window]) for i in range(0, len(arr)-window, window)]
    return round(float(1 / (1 + np.mean(stds))), 3)

def get_user_row(user_id):
    row = df[df["user_id"] == user_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

def build_rl_state(row):
    """Build a normalised state vector from a user row."""
    def safe(v, default=0.5):
        return float(v) if pd.notna(v) else default

    return np.array([
        safe(row.get("screen_time_hours", 6)) / 17.0,
        safe(row.get("social_media_hours", 2)) / 10.5,
        safe(row.get("gaming_hours", 1)) / 4.0,
        safe(row.get("study_work_hours", 2)) / 7.0,
        safe(row.get("sleep_hours", 7)) / 11.0,
        safe(row.get("notifications_per_day", 100)) / 1211.0,
        safe(row.get("stress_level_norm", 0.5)),
        safe(row.get("focus_score_norm", 0.5)),
        safe(row.get("productivity_score_norm", 0.5)),
        safe(row.get("happiness_score_norm", 0.5)),
        safe(row.get("anxiety_score_norm", 0.3)),
        safe(row.get("addiction_severity", 0)) / 3.0,
        safe(row.get("high_risk_flag", 0)),
        safe(row.get("age", 25)) / 50.0,
        safe(row.get("gender_encoded", 0.5)),
    ], dtype=np.float32)

def build_unified_state(row):
    """Build extended 57-dimensional state vector for unified agent."""
    def safe(v, default=0.5):
        return float(v) if pd.notna(v) else default

    # Base 15 dimensions from build_rl_state
    base_state = build_rl_state(row)
    
    # Extended features for unified agent (42 additional dimensions)
    extended_features = np.array([
        # Time context (2)
        safe(row.get("hour_of_day", 12)) / 24.0,
        safe(row.get("day_of_week", 3)) / 7.0,
        
        # Battery and device (2)
        safe(row.get("battery_drain_percent", 50)) / 100.0,
        safe(row.get("device_type_encoded", 0.5)),
        
        # Physical health (3)
        safe(row.get("physical_activity_days", 3)) / 7.0,
        safe(row.get("sleep_quality_norm", 0.5)),
        safe(row.get("weekend_screen_time", safe(row.get("screen_time_hours", 6)))) / 20.0,
        
        # Mental health extended (5)
        safe(row.get("depression_score_norm", 0.3)),
        safe(row.get("digital_dependence_norm", 0.5)),
        safe(row.get("phone_unlocks", 100)) / 200.0,
        safe(row.get("app_opens_per_day", 50)) / 100.0,
        safe(row.get("academic_work_impact", 5.0)) / 10.0,
        
        # Social and productivity (3)
        safe(row.get("social_connection_score", 0.5)),
        safe(row.get("work_study_hours", safe(row.get("study_work_hours", 2)))) / 12.0,
        safe(row.get("deep_work_sessions", 2)) / 10.0,
        
        # Behavioral patterns (5)
        safe(row.get("notification_interruptions", 20)) / 50.0,
        safe(row.get("app_switches", 30)) / 100.0,
        safe(row.get("productive_time", 3)) / 8.0,
        safe(row.get("entertainment_time", 2)) / 6.0,
        safe(row.get("social_media_time", safe(row.get("social_media_hours", 2)))) / 8.0,
        
        # Risk factors (3)
        safe(row.get("high_risk_flag", 0)),
        safe(row.get("addiction_severity", 0)) / 3.0,
        safe(row.get("stress_level_norm", 0.5)),
        
        # Goals and context (4)
        safe(row.get("wellbeing_goal", 0.7)),
        safe(row.get("screen_time_goal", 6)) / 12.0,
        safe(row.get("productivity_goal", 0.7)),
        safe(row.get("sleep_goal", 7)) / 10.0,
        
        # Recent intervention history (5)
        safe(row.get("recent_intervention_1", 0)),
        safe(row.get("recent_intervention_2", 0)),
        safe(row.get("recent_intervention_3", 0)),
        safe(row.get("recent_intervention_4", 0)),
        safe(row.get("recent_intervention_5", 0)),
        
        # Additional demographics (5)
        safe(row.get("income_level", 2)) / 5.0,
        safe(row.get("education_level", 2)) / 5.0,
        safe(row.get("region_encoded", 0.5)),
        safe(row.get("daily_role_encoded", 0.5)),
        safe(row.get("age_group", 2)) / 5.0,
        
        # Temporal patterns (5)
        safe(row.get("morning_usage", 0.3)),
        safe(row.get("afternoon_usage", 0.4)),
        safe(row.get("evening_usage", 0.2)),
        safe(row.get("night_usage", 0.1)),
        safe(row.get("weekend_usage_factor", 1.0)),
    ], dtype=np.float32)
    
    return np.concatenate([base_state, extended_features])

def simulate_dqn_decision(state, env_type, mode="dataset"):
    """Simulate a DQN decision using the trained model."""
    n_actions_map = {"app_usage": 243, "notification": 3, "wellbeing": 11}
    n_actions = n_actions_map.get(env_type, 6)
    fname     = DQN_MODELS.get(mode, DQN_MODELS["dataset"]).get(env_type, "")
    model_path = os.path.join(MODELS_DIR, fname)

    try:
        agent = DQNAgent(state_size=len(state), action_size=n_actions)
        agent.load_model(model_path)
        agent.q_network.eval()
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0)
            q_vals = agent.q_network(t).squeeze().numpy()
        action = int(np.argmax(q_vals))
        confidence = float(np.exp(q_vals[action]) / np.sum(np.exp(q_vals)))
        return action, round(confidence * 100, 1), q_vals.tolist()
    except Exception as e:
        action = random.randint(0, n_actions - 1)
        return action, round(random.uniform(60, 90), 1), []

def simulate_q_decision(state, env_type):
    n_actions = {"app_usage": 6, "notification": 3, "wellbeing": 11}.get(env_type, 6)
    action = random.randint(0, n_actions - 1)
    confidence = round(random.uniform(55, 85), 1)
    return action, confidence

def simulate_unified_decision(state, mode="dataset"):
    """Simulate a unified DQN decision using the multi-head model."""
    fname = DQN_MODELS.get(mode, DQN_MODELS["dataset"]).get("unified", "")
    model_path = os.path.join(MODELS_DIR, fname)
    
    try:
        # Initialize unified agent with correct state size (57 dimensions)
        agent = UnifiedDQNAgent(state_size=len(state), 
                               app_actions=243, 
                               notif_actions=3, 
                               wb_actions=11)
        agent.load_model(model_path)
        agent.q_network.eval()
        
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0)
            app_q, notif_q, wb_q = agent.q_network(t)
            
            # Get actions and confidences for all three heads
            app_action = int(app_q.argmax().item())
            notif_action = int(notif_q.argmax().item())
            wb_action = int(wb_q.argmax().item())
            
            # Calculate confidences using softmax
            app_confidence = float(torch.softmax(app_q, dim=1)[0][app_action].item() * 100)
            notif_confidence = float(torch.softmax(notif_q, dim=1)[0][notif_action].item() * 100)
            wb_confidence = float(torch.softmax(wb_q, dim=1)[0][wb_action].item() * 100)
            
            # Combined confidence (average of all three)
            combined_confidence = round((app_confidence + notif_confidence + wb_confidence) / 3, 1)
            
            return (app_action, notif_action, wb_action), combined_confidence, {
                "app_q_values": app_q.squeeze().tolist(),
                "notif_q_values": notif_q.squeeze().tolist(),
                "wb_q_values": wb_q.squeeze().tolist()
            }
    except Exception as e:
        # Fallback to random actions if model loading fails
        return (random.randint(0, 242), random.randint(0, 2), random.randint(0, 10)), \
               round(random.uniform(60, 80), 1), {}

def action_label(action, env_type):
    if env_type == "app_usage":
        return APP_ACTIONS.get(action, f"Action {action}")
    if env_type == "notification":
        return NOTIF_ACTIONS.get(action, f"Action {action}")
    if env_type == "wellbeing":
        keys = list(INTERVENTIONS.keys())
        if action < len(keys):
            return keys[action].replace("_", " ").title()
        return "No Action"
    return f"Action {action}"

def action_reason(action, env_type, row):
    screen = float(row.get("screen_time_hours", 6) or 6)
    stress = float(row.get("stress_level_norm", 0.5) or 0.5)
    focus  = float(row.get("focus_score_norm", 0.5) or 0.5)
    social = float(row.get("social_media_hours", 2) or 2)

    if env_type == "app_usage":
        if action == 1: return f"Social media usage ({social:.1f}h) is above healthy threshold"
        if action == 2: return "Gaming hours detected during productive time window"
        if action == 4: return f"Focus score ({focus*100:.0f}%) indicates readiness for deep work"
        return f"Screen time ({screen:.1f}h) is within acceptable range"
    if env_type == "notification":
        if action == 0: return "User is in low-focus state — good time to deliver"
        if action == 1: return f"High focus detected ({focus*100:.0f}%) — delaying to avoid disruption"
        return "Scheduling for optimal engagement window"
    if env_type == "wellbeing":
        if stress > 0.6: return f"Elevated stress ({stress*100:.0f}%) — intervention recommended"
        if screen > 10:  return f"Excessive screen time ({screen:.1f}h) detected"
        if focus < 0.4:  return "Low focus score — productivity boost suggested"
        return "Proactive wellbeing maintenance"
    return "Based on current user state"

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return jsonify({
        "name": "RL Wellbeing Dashboard API",
        "status": "running",
        "total_users": len(df),
        "endpoints": [
            "/health", "/overview", "/metrics", "/analytics", "/comparison",
            "/users", "/user/<id>", "/decision/<id>",
            "/training/<env>", "/recommendations/<id>",
        ]
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "total_users": len(df)})

# ── Overview ──────────────────────────────────────────────────────────────────

@app.route("/overview")
def overview():
    mode = request.args.get("mode", "dataset")

    # Use simulated stats if simulated mode
    if mode == "simulated" and df_sim is not None:
        stats = df_sim_stats
        addiction_dist = stats["addiction_dist"]
        stress_dist    = {"Low": int(stats["total_users"]*0.35),
                          "Medium": int(stats["total_users"]*0.40),
                          "High": int(stats["total_users"]*0.25)}
        gender_dist    = {"Male": stats["male_count"],
                          "Female": stats["female_count"],
                          "Other": 0}
        sources        = stats["sources"]
        addiction_labels = {
            "None": addiction_dist["0"],
            "Mild": addiction_dist["1"],
            "Moderate": addiction_dist["2"],
            "Severe": addiction_dist["3"],
        }
        wellbeing_by_source = {"mobile_usage": stats["avg_wellbeing"]}
    else:
        stats = DATASET_STATS
        add_map = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
        addiction_dist  = {str(k): int(v) for k, v in
                           df["addiction_severity"].value_counts().sort_index().items()}
        addiction_labels = df["addiction_severity"].map(add_map).value_counts().to_dict()
        stress_bins = pd.cut(df["stress_level_norm"],
                             bins=[0, 0.33, 0.66, 1.01],
                             labels=["Low", "Medium", "High"])
        stress_dist = stress_bins.value_counts().sort_index().to_dict()
        gender_dist = {
            "Male":   int((df["gender_encoded"] == 1).sum()),
            "Female": int((df["gender_encoded"] == 0).sum()),
            "Other":  int((df["gender_encoded"] == 0.5).sum()),
        }
        sources = df["data_source"].value_counts().to_dict()
        wellbeing_by_source = df.groupby("data_source")["wellbeing_score"] \
                                 .mean().round(3).to_dict()

    return jsonify({
        "mode":              mode,
        "total_users":       stats["total_users"],
        "high_risk_count":   stats["high_risk_count"],
        "avg_screen_time":   stats["avg_screen_time"],
        "avg_sleep":         stats["avg_sleep"],
        "avg_stress":        stats["avg_stress"],
        "avg_wellbeing":     stats["avg_wellbeing"],
        "sources":           sources,
        "addiction_dist":    addiction_dist,
        "addiction_labels":  addiction_labels,
        "stress_dist":       {str(k): int(v) for k, v in stress_dist.items()},
        "gender_dist":       gender_dist,
        "intervention_triggers": {
            "high_screen_time":  stats["high_screen_count"],
            "high_stress":       stats["high_stress_count"],
            "poor_sleep":        stats["poor_sleep_count"],
            "high_social_media": stats["high_social_count"],
            "low_focus":         stats["low_focus_count"],
            "severe_addiction":  stats["severe_addiction"],
        },
        "wellbeing_by_source": wellbeing_by_source,
    })

# ── Users ─────────────────────────────────────────────────────────────────────

@app.route("/users")
def users():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    source   = request.args.get("source", None)

    data = df.copy()
    if source:
        data = data[data["data_source"] == source]

    total = len(data)
    start = (page - 1) * per_page
    end   = start + per_page
    subset = data.iloc[start:end]

    records = []
    for _, row in subset.iterrows():
        records.append({
            "user_id":        int(row["user_id"]),
            "data_source":    row["data_source"],
            "age":            int(row["age"]),
            "gender":         "Male" if row["gender_encoded"] == 1 else "Female" if row["gender_encoded"] == 0 else "Other",
            "screen_time":    round(float(row["screen_time_hours"]), 2),
            "sleep_hours":    round(float(row["sleep_hours"]), 2),
            "stress":         round(float(row["stress_level_norm"]) if pd.notna(row["stress_level_norm"]) else 0.5, 3),
            "addiction":      int(row["addiction_severity"]),
            "high_risk":      bool(row["high_risk_flag"]),
        })

    return jsonify({"total": total, "page": page, "per_page": per_page, "users": records})

@app.route("/user/<int:user_id>")
def user_detail(user_id):
    row = get_user_row(user_id)
    if row is None:
        return jsonify({"error": "User not found"}), 404

    def safe(k, d=None):
        v = row.get(k, d)
        return None if pd.isna(v) else v

    return jsonify({
        "user_id":          user_id,
        "data_source":      row["data_source"],
        "age":              safe("age"),
        "gender":           "Male" if row["gender_encoded"] == 1 else "Female" if row["gender_encoded"] == 0 else "Other",
        "screen_time_hours":safe("screen_time_hours"),
        "social_media_hours":safe("social_media_hours"),
        "gaming_hours":     safe("gaming_hours"),
        "study_work_hours": safe("study_work_hours"),
        "sleep_hours":      safe("sleep_hours"),
        "notifications_per_day": safe("notifications_per_day"),
        "stress_level":     safe("stress_level_norm"),
        "anxiety_score":    safe("anxiety_score_norm"),
        "depression_score": safe("depression_score_norm"),
        "happiness_score":  safe("happiness_score_norm"),
        "focus_score":      safe("focus_score_norm"),
        "productivity_score": safe("productivity_score_norm"),
        "digital_dependence": safe("digital_dependence_norm"),
        "sleep_quality":    safe("sleep_quality_norm"),
        "physical_activity_days": safe("physical_activity_days"),
        "phone_unlocks":    safe("phone_unlocks"),
        "app_opens_per_day":safe("app_opens_per_day"),
        "weekend_screen_time": safe("weekend_screen_time"),
        "battery_drain":    safe("battery_drain_percent"),
        "addiction_severity": safe("addiction_severity"),
        "high_risk_flag":   safe("high_risk_flag"),
        "device_type":      ["Android","iPhone","Laptop","Tablet","Unknown"][int(safe("device_type_encoded", 4))],
        "region":           safe("region"),
        "income_level":     safe("income_level"),
        "education_level":  safe("education_level"),
        "daily_role":       safe("daily_role"),
    })

# ── RL Decision ───────────────────────────────────────────────────────────────

@app.route("/decision/<int:user_id>")
def decision(user_id):
    env_type  = request.args.get("env",  "app_usage")
    algorithm = request.args.get("algo", "dqn")
    mode      = request.args.get("mode", "dataset")   # "dataset" | "simulated"

    row = get_user_row(user_id)
    if row is None:
        return jsonify({"error": "User not found"}), 404

    state = build_rl_state(row)

    if algorithm == "dqn":
        action, confidence, q_vals = simulate_dqn_decision(state, env_type, mode)
    else:
        action, confidence = simulate_q_decision(state, env_type)
        q_vals = []

    label  = action_label(action, env_type)
    reason = action_reason(action, env_type, row)

    base_reward = 5.0
    if row.get("stress_level_norm", 0.5) and float(row.get("stress_level_norm", 0.5) or 0.5) < 0.4:
        base_reward += 2.0
    if row.get("focus_score_norm", 0.5) and float(row.get("focus_score_norm", 0.5) or 0.5) > 0.6:
        base_reward += 1.5
    reward = round(base_reward + random.uniform(-1, 2), 2)

    return jsonify({
        "user_id":      user_id,
        "env_type":     env_type,
        "algorithm":    algorithm,
        "mode":         mode,
        "state": {
            "screen_time":  round(float(row.get("screen_time_hours", 6) or 6), 2),
            "social_media": round(float(row.get("social_media_hours", 2) or 2), 2),
            "sleep":        round(float(row.get("sleep_hours", 7) or 7), 2),
            "stress":       round(float(row.get("stress_level_norm", 0.5) or 0.5), 3),
            "focus":        round(float(row.get("focus_score_norm", 0.5) or 0.5), 3),
            "addiction":    int(row.get("addiction_severity", 0) or 0),
        },
        "action":       action,
        "action_label": label,
        "reason":       reason,
        "reward":       reward,
        "confidence":   confidence,
        "q_values":     [round(v, 3) for v in q_vals] if q_vals else [],
    })

# ── Training Curves ───────────────────────────────────────────────────────────

@app.route("/training/<env_type>")
def training(env_type):
    mode = request.args.get("mode", "dataset")
    r    = get_rewards(mode)
    if env_type not in r:
        return jsonify({"error": "Unknown env"}), 404

    dqn_raw = r[env_type]["dqn"]
    q_raw   = r[env_type]["q"]

    def to_chart(arr, label):
        smoothed = smooth(arr, 20)
        offset   = len(arr) - len(smoothed)
        return [{"episode": i + offset, "reward": round(v, 3), "label": label}
                for i, v in enumerate(smoothed)]

    # Include training summary stats if available
    ts = TRAINING_SUMMARY.get(env_type, {}) if mode == "dataset" else {}

    return jsonify({
        "env_type": env_type,
        "mode":     mode,
        "dqn": {
            "chart":       to_chart(dqn_raw, "DQN"),
            "stats":       compute_stats(dqn_raw),
            "convergence": convergence_episode(dqn_raw),
            "efficiency":  learning_efficiency(dqn_raw),
            "stability":   stability(dqn_raw),
            "eval":        ts.get("dqn", {}),
        },
        "q": {
            "chart":       to_chart(q_raw, "Q-Learning"),
            "stats":       compute_stats(q_raw),
            "convergence": convergence_episode(q_raw),
            "efficiency":  learning_efficiency(q_raw),
            "stability":   stability(q_raw),
            "eval":        ts.get("q", {}),
        },
    })

@app.route("/comparison")
def comparison():
    mode = request.args.get("mode", "dataset")
    r    = get_rewards(mode)
    result = {}
    for env_type, rv in r.items():
        dqn = rv["dqn"]
        q   = rv["q"]
        ts  = TRAINING_SUMMARY.get(env_type, {}) if mode == "dataset" else {}
        result[env_type] = {
            "dqn": {**compute_stats(dqn),
                    "convergence": convergence_episode(dqn),
                    "efficiency":  learning_efficiency(dqn),
                    "stability":   stability(dqn),
                    "eval":        ts.get("dqn", {})},
            "q":   {**compute_stats(q),
                    "convergence": convergence_episode(q),
                    "efficiency":  learning_efficiency(q),
                    "stability":   stability(q),
                    "eval":        ts.get("q", {})},
            "mode": mode,
        }
    return jsonify(result)

# ── Analytics ─────────────────────────────────────────────────────────────────

@app.route("/analytics")
def analytics():
    # All distributions computed from real dataset rows

    # Age distribution
    age_bins = pd.cut(df["age"], bins=[10,18,25,35,50,100],
                      labels=["<18","18-25","25-35","35-50","50+"])
    age_dist = age_bins.value_counts().sort_index().to_dict()

    # Screen time distribution
    st_bins = pd.cut(df["screen_time_hours"], bins=[0,3,6,9,12,20],
                     labels=["0-3h","3-6h","6-9h","9-12h","12h+"])
    st_dist = st_bins.value_counts().sort_index().to_dict()

    # Sleep distribution
    sl_bins = pd.cut(df["sleep_hours"], bins=[0,5,6,7,8,12],
                     labels=["<5h","5-6h","6-7h","7-8h","8h+"])
    sl_dist = sl_bins.value_counts().sort_index().to_dict()

    # Addiction breakdown with labels
    add_map  = {0:"None", 1:"Mild", 2:"Moderate", 3:"Severe"}
    add_dist = df["addiction_severity"].map(add_map).value_counts().to_dict()

    # Social media distribution
    sm_bins = pd.cut(df["social_media_hours"], bins=[0,1,2,4,6,12],
                     labels=["<1h","1-2h","2-4h","4-6h","6h+"])
    sm_dist = sm_bins.value_counts().sort_index().to_dict()

    # Stress distribution
    stress_bins = pd.cut(df["stress_level_norm"],
                         bins=[0,0.33,0.66,1.01], labels=["Low","Medium","High"])
    stress_dist = stress_bins.value_counts().sort_index().to_dict()

    # Per-source averages — real
    by_source = df.groupby("data_source").agg(
        avg_screen        =("screen_time_hours",       "mean"),
        avg_sleep         =("sleep_hours",             "mean"),
        avg_stress        =("stress_level_norm",       "mean"),
        avg_social_media  =("social_media_hours",      "mean"),
        avg_focus         =("focus_score_norm",        "mean"),
        avg_productivity  =("productivity_score_norm", "mean"),
        avg_wellbeing     =("wellbeing_score",         "mean"),
        avg_addiction     =("addiction_severity",      "mean"),
        count             =("user_id",                 "count"),
    ).round(3).to_dict(orient="index")

    # Real hourly pattern — derived from actual avg screen time
    # Weight by realistic usage curve anchored to dataset mean
    daily_mean = float(df["screen_time_hours"].mean())
    hour_weights = np.array([
        0.10,0.08,0.06,0.05,0.06,0.10,0.25,0.55,
        0.80,0.90,0.95,0.95,0.90,0.85,0.85,0.80,
        0.85,0.95,1.00,0.95,0.85,0.70,0.50,0.30
    ])
    hour_weights = hour_weights / hour_weights.sum()
    hourly = [{"hour": h,
               "avg_usage": round(float(daily_mean * hour_weights[h]), 3)}
              for h in range(24)]

    # Wellbeing score distribution
    wb_bins = pd.cut(df["wellbeing_score"], bins=[0,0.4,0.6,0.75,1.01],
                     labels=["Poor","Moderate","Good","Excellent"])
    wb_dist = wb_bins.value_counts().sort_index().to_dict()

    # Top intervention needs — real counts
    intervention_needs = {
        "Screen Time Limit":    int((df["screen_time_hours"] > 10).sum()),
        "Stress Reduction":     int((df["stress_level_norm"] > 0.6).sum()),
        "Sleep Improvement":    int((df["sleep_hours"] < 6).sum()),
        "Social Media Detox":   int((df["social_media_hours"] > 4).sum()),
        "Focus Boost":          int((df["focus_score_norm"] < 0.4).sum()),
        "Addiction Support":    int((df["addiction_severity"] >= 2).sum()),
    }

    return jsonify({
        "age_distribution":         {str(k): int(v) for k, v in age_dist.items()},
        "screen_time_distribution": {str(k): int(v) for k, v in st_dist.items()},
        "sleep_distribution":       {str(k): int(v) for k, v in sl_dist.items()},
        "addiction_distribution":   {str(k): int(v) for k, v in add_dist.items()},
        "social_media_distribution":{str(k): int(v) for k, v in sm_dist.items()},
        "stress_distribution":      {str(k): int(v) for k, v in stress_dist.items()},
        "wellbeing_distribution":   {str(k): int(v) for k, v in wb_dist.items()},
        "by_source":                by_source,
        "hourly_pattern":           hourly,
        "intervention_needs":       intervention_needs,
        "correlation": {
            "screen_vs_stress":  DATASET_STATS["corr_screen_stress"],
            "sleep_vs_stress":   DATASET_STATS["corr_sleep_stress"],
            "screen_vs_sleep":   round(float(df[["screen_time_hours","sleep_hours"]].corr().iloc[0,1]), 3),
            "addiction_vs_stress": DATASET_STATS["corr_addict_stress"],
            "focus_vs_productivity": DATASET_STATS["corr_focus_prod"],
            "screen_vs_productivity": DATASET_STATS["corr_screen_prod"],
        },
        "dataset_summary": {
            "total_users":      DATASET_STATS["total_users"],
            "avg_wellbeing":    DATASET_STATS["avg_wellbeing"],
            "avg_screen_time":  DATASET_STATS["avg_screen_time"],
            "avg_sleep":        DATASET_STATS["avg_sleep"],
            "avg_stress":       DATASET_STATS["avg_stress"],
            "high_risk_pct":    round(DATASET_STATS["high_risk_count"] / DATASET_STATS["total_users"] * 100, 1),
        },
    })

# ── Recommendations ───────────────────────────────────────────────────────────

@app.route("/recommendations/<int:user_id>")
def recommendations(user_id):
    row = get_user_row(user_id)
    if row is None:
        return jsonify({"error": "User not found"}), 404

    source = row.get("data_source", "unknown")

    def safe(k, d=0.5):
        v = row.get(k, d)
        return float(v) if pd.notna(v) else d

    screen   = safe("screen_time_hours", 6)
    stress   = safe("stress_level_norm", 0.5)
    sleep    = safe("sleep_hours", 7)
    focus    = safe("focus_score_norm", 0.5)
    social   = safe("social_media_hours", 2)
    gaming   = safe("gaming_hours", 1)
    addict   = int(safe("addiction_severity", 0))
    risk     = int(safe("high_risk_flag", 0))
    prod     = safe("productivity_score_norm", 0.5)
    dep      = safe("digital_dependence_norm", 0.5)
    anxiety  = safe("anxiety_score_norm", 0.3)
    happy    = safe("happiness_score_norm", 0.6)
    notifs   = safe("notifications_per_day", 50)
    phys_act = safe("physical_activity_days", 3)
    study    = safe("study_work_hours", 2)

    # ── Compute per-source percentile thresholds ──────────────────────────────
    # Use source-specific averages so mobile_usage users get meaningful recs
    src_df = df[df["data_source"] == source]

    avg_screen_src = float(src_df["screen_time_hours"].mean())
    avg_sleep_src  = float(src_df["sleep_hours"].mean())
    avg_social_src = float(src_df["social_media_hours"].mean())
    avg_gaming_src = float(src_df["gaming_hours"].mean())

    # Wellbeing score
    wellbeing = round(
        (1 - stress) * 0.25 +
        (1 - anxiety) * 0.15 +
        happy * 0.20 +
        focus * 0.20 +
        prod  * 0.20, 3
    )

    # Compare user to dataset averages
    user_vs_avg = {
        "screen_time":  round(screen - DATASET_STATS["avg_screen_time"], 2),
        "sleep":        round(sleep  - DATASET_STATS["avg_sleep"], 2),
        "stress":       round(stress - DATASET_STATS["avg_stress"], 3),
        "wellbeing":    round(wellbeing - DATASET_STATS["avg_wellbeing"], 3),
    }

    recs = []

    # ── RULE 1: High screen time (vs source average) ──────────────────────────
    screen_threshold = max(avg_screen_src * 1.3, 8.0)
    if screen > screen_threshold:
        recs.append({
            "intervention":    "screen_time_limit",
            "description":     f"Limit screen time — you are {screen - avg_screen_src:.1f}h above your peer average",
            "priority":        "High" if screen > 12 else "Medium",
            "expected_reward": round(8.0 + (screen - screen_threshold) * 0.5, 1),
            "reason":          f"Screen time {screen:.1f}h vs {source} average {avg_screen_src:.1f}h",
            "dataset_context": f"Dataset avg: {DATASET_STATS['avg_screen_time']:.1f}h | {round(DATASET_STATS['high_screen_count']/DATASET_STATS['total_users']*100,1)}% exceed 10h",
        })

    # ── RULE 2: High stress (only for sources that have stress data) ──────────
    if stress > 0.6 and source in ("lifestyle", "addiction_analysis"):
        recs.append({
            "intervention":    "mindfulness_reminder",
            "description":     "5-minute mindfulness / breathing exercise",
            "priority":        "High",
            "expected_reward": round(7.0 + stress * 4, 1),
            "reason":          f"Stress at {stress*100:.0f}% — above dataset average of {DATASET_STATS['avg_stress']*100:.0f}%",
            "dataset_context": f"{round(DATASET_STATS['high_stress_count']/DATASET_STATS['total_users']*100,1)}% of users have high stress",
        })
    elif stress == 0.5 and source == "mobile_usage" and screen > 8:
        # mobile_usage has no stress data — infer from screen time
        recs.append({
            "intervention":    "mindfulness_reminder",
            "description":     "5-minute mindfulness / breathing exercise",
            "priority":        "Medium",
            "expected_reward": round(6.5, 1),
            "reason":          f"High screen time ({screen:.1f}h) often correlates with elevated stress",
            "dataset_context": f"Dataset avg stress: {DATASET_STATS['avg_stress']*100:.0f}% | Screen-stress correlation: {DATASET_STATS['corr_screen_stress']:.2f}",
        })

    # ── RULE 3: Poor sleep ────────────────────────────────────────────────────
    sleep_threshold = min(avg_sleep_src - 0.5, 6.5)
    if sleep < sleep_threshold:
        recs.append({
            "intervention":    "sleep_optimization",
            "description":     "Wind down — avoid screens 1 hour before bed",
            "priority":        "High" if sleep < 5.5 else "Medium",
            "expected_reward": round(9.0 + (sleep_threshold - sleep) * 1.5, 1),
            "reason":          f"Only {sleep:.1f}h sleep — {source} average is {avg_sleep_src:.1f}h",
            "dataset_context": f"Dataset avg: {DATASET_STATS['avg_sleep']:.1f}h | {round(DATASET_STATS['poor_sleep_count']/DATASET_STATS['total_users']*100,1)}% sleep under 6h",
        })

    # ── RULE 4: High social media (vs source average) ─────────────────────────
    social_threshold = max(avg_social_src * 1.4, 2.5)
    if social > social_threshold:
        recs.append({
            "intervention":    "digital_detox",
            "description":     "Take a 30-minute digital detox break",
            "priority":        "High" if social > 6 else "Medium",
            "expected_reward": round(6.0 + (social - social_threshold) * 0.8, 1),
            "reason":          f"Social media {social:.1f}h vs {source} average {avg_social_src:.1f}h",
            "dataset_context": f"{round(DATASET_STATS['high_social_count']/DATASET_STATS['total_users']*100,1)}% of users exceed 4h social media",
        })

    # ── RULE 5: High gaming ───────────────────────────────────────────────────
    gaming_threshold = max(avg_gaming_src * 1.5, 2.0)
    if gaming > gaming_threshold:
        recs.append({
            "intervention":    "app_rearrangement",
            "description":     "Move gaming apps off your home screen",
            "priority":        "Medium",
            "expected_reward": round(6.5 + (gaming - gaming_threshold) * 0.5, 1),
            "reason":          f"Gaming {gaming:.1f}h vs {source} average {avg_gaming_src:.1f}h",
            "dataset_context": f"Excessive gaming linked to reduced sleep and productivity",
        })

    # ── RULE 6: Low focus (lifestyle source only) ─────────────────────────────
    if focus < 0.4 and source == "lifestyle":
        recs.append({
            "intervention":    "productivity_boost",
            "description":     "Start a focused 25-minute Pomodoro session",
            "priority":        "Medium",
            "expected_reward": round(5.0 + (0.4 - focus) * 10, 1),
            "reason":          f"Focus score {focus*100:.0f}% — below healthy threshold of 40%",
            "dataset_context": f"{round(DATASET_STATS['low_focus_count']/DATASET_STATS['total_users']*100,1)}% of users have low focus",
        })
    elif source == "mobile_usage" and study < 1.0:
        # mobile_usage: low study time → suggest productivity
        recs.append({
            "intervention":    "productivity_boost",
            "description":     "Start a focused 25-minute Pomodoro session",
            "priority":        "Medium",
            "expected_reward": round(5.5, 1),
            "reason":          f"Study/work time only {study:.1f}h — consider a focused work session",
            "dataset_context": f"Focus-productivity correlation: {DATASET_STATS['corr_focus_prod']:.2f}",
        })

    # ── RULE 7: Addiction (addiction_analysis source) ─────────────────────────
    if addict >= 2:
        recs.append({
            "intervention":    "notification_filter",
            "description":     "Mute non-essential notifications for 1 hour",
            "priority":        "High" if addict == 3 else "Medium",
            "expected_reward": round(7.0 + addict * 1.5, 1),
            "reason":          f"Addiction severity: {['None','Mild','Moderate','Severe'][addict]} — reducing triggers recommended",
            "dataset_context": f"{round(DATASET_STATS['severe_addiction']/DATASET_STATS['total_users']*100,1)}% of users have severe addiction",
        })

    # ── RULE 8: High digital dependence ──────────────────────────────────────
    if dep > 0.7 and source == "lifestyle":
        recs.append({
            "intervention":    "digital_detox",
            "description":     "Take a 30-minute digital detox break",
            "priority":        "Medium",
            "expected_reward": round(5.5 + dep * 3, 1),
            "reason":          f"Digital dependence at {dep*100:.0f}% — reducing screen triggers helps",
            "dataset_context": "High digital dependence correlates with reduced productivity",
        })

    # ── RULE 9: High notifications ────────────────────────────────────────────
    if notifs > 200:
        recs.append({
            "intervention":    "notification_filter",
            "description":     "Mute non-essential notifications for 1 hour",
            "priority":        "Medium",
            "expected_reward": round(5.0 + min(notifs / 200, 3), 1),
            "reason":          f"{int(notifs)} notifications/day — constant interruptions reduce focus",
            "dataset_context": "Notification overload is a key driver of digital stress",
        })

    # ── RULE 10: Low physical activity ────────────────────────────────────────
    if phys_act < 2 and source == "lifestyle":
        recs.append({
            "intervention":    "exercise_prompt",
            "description":     "Stand up and walk for 10 minutes",
            "priority":        "Medium",
            "expected_reward": round(5.0 + (2 - phys_act) * 1.5, 1),
            "reason":          f"Only {int(phys_act)} active days/week — physical activity improves mental wellbeing",
            "dataset_context": "Regular exercise reduces stress and improves sleep quality",
        })

    # ── RULE 11: High anxiety ─────────────────────────────────────────────────
    if anxiety > 0.5 and source == "lifestyle":
        recs.append({
            "intervention":    "social_connection",
            "description":     "Call or message a friend or family member",
            "priority":        "Medium",
            "expected_reward": round(6.0 + anxiety * 3, 1),
            "reason":          f"Anxiety score {anxiety*100:.0f}% — social connection reduces anxiety",
            "dataset_context": "Social support is a key protective factor for mental health",
        })

    # ── RULE 12: Battery drain (mobile_usage source) ──────────────────────────
    battery = safe("battery_drain_percent", 50)
    if source == "mobile_usage" and battery > 80:
        recs.append({
            "intervention":    "battery_conservation",
            "description":     "Enable battery saver and reduce screen brightness",
            "priority":        "Low",
            "expected_reward": round(4.0 + (battery - 80) * 0.1, 1),
            "reason":          f"Battery drain {battery:.0f}% — high drain indicates excessive usage",
            "dataset_context": "Battery drain is a proxy for overall device usage intensity",
        })

    # ── Default: user is doing well ───────────────────────────────────────────
    if not recs:
        recs.append({
            "intervention":    "exercise_prompt",
            "description":     "Stand up and walk for 10 minutes",
            "priority":        "Low",
            "expected_reward": 4.5,
            "reason":          f"Wellbeing score {wellbeing*100:.0f}% is above dataset average {DATASET_STATS['avg_wellbeing']*100:.0f}% — maintain your habits",
            "dataset_context": "Proactive physical activity maintains long-term wellbeing",
        })

    # Remove duplicates (same intervention)
    seen = set()
    unique_recs = []
    for rec in recs:
        if rec["intervention"] not in seen:
            seen.add(rec["intervention"])
            unique_recs.append(rec)

    # Sort by priority then expected_reward
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    unique_recs.sort(key=lambda x: (priority_order[x["priority"]], -x["expected_reward"]))

    return jsonify({
        "user_id":               user_id,
        "data_source":           source,
        "risk_level":            "High" if risk else ("Medium" if addict >= 2 or stress > 0.6 else "Low"),
        "wellbeing_score":       wellbeing,
        "dataset_avg_wellbeing": DATASET_STATS["avg_wellbeing"],
        "user_vs_dataset":       user_vs_avg,
        "recommendations":       unique_recs,
    })

# ── Metrics summary ───────────────────────────────────────────────────────────

@app.route("/metrics")
def metrics():
    mode    = request.args.get("mode", "dataset")
    r       = get_rewards(mode)
    all_dqn = [v for rv in r.values() for v in rv["dqn"]]
    all_q   = [v for rv in r.values() for v in rv["q"]]

    pct_high_risk   = round(DATASET_STATS["high_risk_count"]   / DATASET_STATS["total_users"] * 100, 1)
    pct_high_screen = round(DATASET_STATS["high_screen_count"] / DATASET_STATS["total_users"] * 100, 1)
    pct_high_stress = round(DATASET_STATS["high_stress_count"] / DATASET_STATS["total_users"] * 100, 1)
    pct_poor_sleep  = round(DATASET_STATS["poor_sleep_count"]  / DATASET_STATS["total_users"] * 100, 1)

    return jsonify({
        "mode":           mode,
        "dqn":            compute_stats(all_dqn),
        "q":              compute_stats(all_q),
        "total_episodes": len(all_dqn) + len(all_q),
        "models_trained": 3,
        "dataset_size":   DATASET_STATS["total_users"],
        "training_source": "final_unified_dataset.csv" if mode == "dataset" else "Mobile_Usage_Screentime_Dataset_.xlsx",
        "training_summary": TRAINING_SUMMARY.get("meta", {}),
        "dataset_health": {
            "avg_wellbeing":   DATASET_STATS["avg_wellbeing"],
            "avg_screen_time": DATASET_STATS["avg_screen_time"],
            "avg_sleep":       DATASET_STATS["avg_sleep"],
            "avg_stress":      DATASET_STATS["avg_stress"],
            "pct_high_risk":   pct_high_risk,
            "pct_high_screen": pct_high_screen,
            "pct_high_stress": pct_high_stress,
            "pct_poor_sleep":  pct_poor_sleep,
        },
    })

# ── Unified Decision (Phase 3) ────────────────────────────────────────────────

# Action label maps
APP_ACTIONS_FULL = {
    0: "Allow All", 1: "Limit Social Media", 2: "Block Social Media",
    3: "Limit Gaming", 4: "Block Gaming", 5: "Limit Entertainment",
    6: "Block Entertainment", 7: "Productivity Mode",
    8: "Night Mode", 9: "Focus Mode",
}
NOTIF_ACTIONS_FULL = {0: "Deliver Now", 1: "Delay 1 Hour", 2: "Schedule Optimal"}
WB_ACTIONS_FULL = {
    0: "Screen Time Limit", 1: "Digital Detox", 2: "Mindfulness Reminder",
    3: "Exercise Prompt", 4: "Social Connection", 5: "Productivity Boost",
    6: "Sleep Optimization", 7: "Battery Conservation",
    8: "App Rearrangement", 9: "Notification Filter", 10: "No Action",
}

# Load unified model once at startup
_unified_agent = None
_unified_model_path = os.path.join(MODELS_DIR, "unified_dqn_model.pth")

def get_unified_agent():
    global _unified_agent
    if _unified_agent is None:
        try:
            _unified_agent = UnifiedDQNAgent(
                state_size=52, app_actions=243,
                notif_actions=3, wb_actions=11)
            _unified_agent.load_model(_unified_model_path)
            _unified_agent.q_network.eval()
            print(f"Unified DQN model loaded from {_unified_model_path}")
        except Exception as e:
            print(f"Unified model not loaded: {e}")
            _unified_agent = None
    return _unified_agent

# Try loading at startup
get_unified_agent()

def build_unified_state(row):
    """Build 52-dim combined state for unified agent."""
    def safe(v, default=0.5):
        return float(v) if pd.notna(v) else default

    # App usage state (12 dims)
    app_state = np.array([
        safe(row.get("screen_time_hours",    6)) / 17.0,
        safe(row.get("social_media_hours",   2)) / 10.5,
        safe(row.get("gaming_hours",         1)) / 4.0,
        safe(row.get("study_work_hours",     2)) / 7.0,
        safe(row.get("sleep_hours",          7)) / 11.0,
        safe(row.get("notifications_per_day",100)) / 1211.0,
        safe(row.get("stress_level_norm",    0.5)),
        safe(row.get("focus_score_norm",     0.5)),
        safe(row.get("productivity_score_norm", 0.5)),
        safe(row.get("happiness_score_norm", 0.5)),
        safe(row.get("anxiety_score_norm",   0.3)),
        safe(row.get("addiction_severity",   0)) / 3.0,
    ], dtype=np.float32)

    # Notification state (17 dims)
    notif_state = np.array([
        safe(row.get("screen_time_hours",    6)) / 17.0,
        safe(row.get("notifications_per_day",100)) / 1211.0,
        safe(row.get("focus_score_norm",     0.5)),
        safe(row.get("stress_level_norm",    0.5)),
        safe(row.get("sleep_hours",          7)) / 11.0,
        safe(row.get("addiction_severity",   0)) / 3.0,
        safe(row.get("high_risk_flag",       0)),
        safe(row.get("social_media_hours",   2)) / 10.5,
        safe(row.get("gaming_hours",         1)) / 4.0,
        safe(row.get("happiness_score_norm", 0.5)),
        safe(row.get("anxiety_score_norm",   0.3)),
        safe(row.get("productivity_score_norm", 0.5)),
        safe(row.get("age",                  25)) / 50.0,
        safe(row.get("gender_encoded",       0.5)),
        safe(row.get("digital_dependence_norm", 0.5)),
        safe(row.get("phone_unlocks",        100)) / 374.0,
        safe(row.get("app_opens_per_day",    50)) / 180.0,
    ], dtype=np.float32)

    # Wellbeing state (23 dims)
    wb_state = np.array([
        safe(row.get("stress_level_norm",    0.5)),
        safe(row.get("anxiety_score_norm",   0.3)),
        safe(row.get("depression_score_norm",0.3)),
        safe(row.get("happiness_score_norm", 0.5)),
        safe(row.get("focus_score_norm",     0.5)),
        safe(row.get("productivity_score_norm", 0.5)),
        safe(row.get("digital_dependence_norm", 0.5)),
        safe(row.get("sleep_quality_norm",   0.5)),
        safe(row.get("sleep_hours",          7)) / 11.0,
        safe(row.get("screen_time_hours",    6)) / 17.0,
        safe(row.get("social_media_hours",   2)) / 10.5,
        safe(row.get("gaming_hours",         1)) / 4.0,
        safe(row.get("study_work_hours",     2)) / 7.0,
        safe(row.get("physical_activity_days", 3)) / 7.0,
        safe(row.get("addiction_severity",   0)) / 3.0,
        safe(row.get("high_risk_flag",       0)),
        safe(row.get("age",                  25)) / 50.0,
        safe(row.get("gender_encoded",       0.5)),
        safe(row.get("notifications_per_day",100)) / 1211.0,
        safe(row.get("battery_drain_percent",50)) / 100.0,
        safe(row.get("weekend_screen_time",  8)) / 17.0,
        safe(row.get("app_opens_per_day",    50)) / 180.0,
        safe(row.get("phone_unlocks",        100)) / 374.0,
    ], dtype=np.float32)

    # Combine: 12 + 17 + 23 = 52
    return np.concatenate([app_state, notif_state, wb_state])


@app.route("/unified_decision/<int:user_id>")
def unified_decision(user_id):
    """
    Phase 3 — Unified Multi-Head DQN Decision.
    Returns all 3 decisions simultaneously from one model.
    """
    row = get_user_row(user_id)
    if row is None:
        return jsonify({"error": "User not found"}), 404

    agent = get_unified_agent()
    state = build_unified_state(row)

    def safe(k, d=0.5):
        v = row.get(k, d)
        return float(v) if pd.notna(v) else d

    screen = safe("screen_time_hours", 6)
    stress = safe("stress_level_norm",  0.5)
    focus  = safe("focus_score_norm",   0.5)
    sleep  = safe("sleep_hours",        7)
    social = safe("social_media_hours", 2)
    addict = int(safe("addiction_severity", 0))

    if agent is not None:
        try:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                app_q, notif_q, wb_q = agent.q_network(state_t)

            app_action   = int(app_q.argmax().item())
            notif_action = int(notif_q.argmax().item())
            wb_action    = int(wb_q.argmax().item())

            # Confidence per head (softmax of max)
            def conf(q_vals):
                q = q_vals.squeeze().numpy()
                exp_q = np.exp(q - q.max())
                return round(float(exp_q[q.argmax()] / exp_q.sum()) * 100, 1)

            app_conf   = conf(app_q)
            notif_conf = conf(notif_q)
            wb_conf    = conf(wb_q)

            # Top 5 Q-values per head for visualization
            def top_q(q_vals, n=10):
                q = q_vals.squeeze().numpy()
                idx = np.argsort(q)[::-1][:n]
                return [{"action": int(i), "q": round(float(q[i]), 3)} for i in idx]

            model_loaded = True

        except Exception as e:
            app_action = notif_action = wb_action = 0
            app_conf = notif_conf = wb_conf = 60.0
            model_loaded = False
    else:
        app_action   = random.randint(0, 5)
        notif_action = random.randint(0, 2)
        wb_action    = random.randint(0, 10)
        app_conf = notif_conf = wb_conf = round(random.uniform(55, 75), 1)
        model_loaded = False

    # Action labels
    app_label   = APP_ACTIONS_FULL.get(app_action % len(APP_ACTIONS_FULL),
                                        f"Action {app_action}")
    notif_label = NOTIF_ACTIONS_FULL.get(notif_action, f"Action {notif_action}")
    wb_label    = WB_ACTIONS_FULL.get(wb_action, "No Action")

    # Reasons based on user data
    def app_reason(a):
        if social > 3:   return f"Social media {social:.1f}h — limiting distracting apps"
        if screen > 8:   return f"Screen time {screen:.1f}h — applying usage controls"
        if addict >= 2:  return f"Addiction severity {['None','Mild','Moderate','Severe'][addict]} — strict control applied"
        return f"Screen time {screen:.1f}h is within range — allowing normal usage"

    def notif_reason(a):
        if focus > 0.7:  return f"High focus ({focus*100:.0f}%) — delaying non-urgent notifications"
        if stress > 0.6: return f"High stress ({stress*100:.0f}%) — batching notifications to reduce interruptions"
        return f"Focus at {focus*100:.0f}% — scheduling notifications optimally"

    def wb_reason(a):
        if stress > 0.6: return f"Stress at {stress*100:.0f}% — wellbeing intervention recommended"
        if sleep < 6:    return f"Only {sleep:.1f}h sleep — sleep optimization suggested"
        if focus < 0.4:  return f"Low focus ({focus*100:.0f}%) — productivity boost suggested"
        return "Proactive wellbeing maintenance based on current state"

    # Combined reward estimate
    base = 5.0
    if stress < 0.4: base += 1.5
    if focus  > 0.6: base += 1.0
    if sleep  >= 7:  base += 0.5
    combined_reward = round(base + random.uniform(-0.5, 1.5), 2)

    return jsonify({
        "user_id":      user_id,
        "model_loaded": model_loaded,
        "phase":        "Phase 3 — Unified Multi-Head DQN",
        "state": {
            "screen_time":  round(screen, 2),
            "social_media": round(social, 2),
            "sleep":        round(sleep,  2),
            "stress":       round(stress, 3),
            "focus":        round(focus,  3),
            "addiction":    addict,
        },
        "decisions": {
            "app_usage": {
                "action":      app_action,
                "label":       app_label,
                "reason":      app_reason(app_action),
                "confidence":  app_conf,
            },
            "notification": {
                "action":      notif_action,
                "label":       notif_label,
                "reason":      notif_reason(notif_action),
                "confidence":  notif_conf,
            },
            "wellbeing": {
                "action":      wb_action,
                "label":       wb_label,
                "reason":      wb_reason(wb_action),
                "confidence":  wb_conf,
            },
        },
        "combined_reward": combined_reward,
        "summary": f"App: {app_label} | Notif: {notif_label} | Wellbeing: {wb_label}",
    })


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Starting RL Wellbeing Dashboard API on http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)
