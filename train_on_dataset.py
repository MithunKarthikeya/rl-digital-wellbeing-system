"""
train_on_dataset.py
====================
Trains Q-Learning and DQN on all 3 RL environments using
final_unified_dataset.csv (11,120 real users, 0 NaN).

Saves to results/models/:
  dataset_app_usage_dqn_model.pth
  dataset_app_usage_dqn_rewards.npy
  dataset_app_usage_q_rewards.npy
  dataset_notification_dqn_model.pth
  dataset_notification_dqn_rewards.npy
  dataset_notification_q_rewards.npy
  dataset_wellbeing_dqn_model.pth
  dataset_wellbeing_dqn_rewards.npy
  dataset_wellbeing_q_rewards.npy
  training_summary.json
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.rl_algorithms import QLearningAgent, DQNAgent
from app_usage_control.app_usage_environment import AppUsageControlEnvironment
from notification_scheduling.notification_environment import NotificationSchedulingEnvironment
from wellbeing_optimization.wellbeing_environment import DigitalWellbeingOptimizationEnvironment

MODELS_DIR  = os.path.join(ROOT, "results", "models")
PLOTS_DIR   = os.path.join(ROOT, "results", "plots")
REPORTS_DIR = os.path.join(ROOT, "results", "reports")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────

def load_dataset_profiles():
    """Convert final_unified_dataset.csv rows into user profile dicts."""
    path = os.path.join(ROOT, "results", "data", "final_unified_dataset.csv")
    df   = pd.read_csv(path)
    print(f"✅ Loaded dataset: {len(df):,} users, {df.isnull().sum().sum()} NaN")

    profiles = {}
    for _, row in df.iterrows():
        uid = int(row["user_id"])

        # Derive usage_pattern and screen_time_category for env compatibility
        screen = float(row["screen_time_hours"])
        social = float(row["social_media_hours"])
        gaming = float(row["gaming_hours"])
        study  = float(row["study_work_hours"])
        total  = max(screen, 0.1)

        if social / total > 0.4:
            usage_pattern = "social_media_heavy"
        elif gaming / total > 0.3:
            usage_pattern = "gaming_heavy"
        elif study / total > 0.4:
            usage_pattern = "productive"
        else:
            usage_pattern = "balanced"

        if screen < 4:
            screen_cat = "low"
        elif screen < 8:
            screen_cat = "moderate"
        else:
            screen_cat = "high"

        profiles[uid] = {
            # Core identifiers
            "user_id":                  uid,
            "data_source":              row["data_source"],
            # Demographics
            "age":                      int(row["age"]),
            "gender":                   "Male" if row["gender_encoded"] == 1 else "Female",
            # Usage
            "screen_time_hours":        screen,
            "social_media_hours":       social,
            "gaming_hours":             gaming,
            "study_work_hours":         study,
            "sleep_hours":              float(row["sleep_hours"]),
            "notifications_per_day":    int(row["notifications_per_day"]),
            # Normalised scores (0-1)
            "stress_level":             float(row["stress_level_norm"]),
            "anxiety_score":            float(row["anxiety_score_norm"]),
            "depression_score":         float(row["depression_score_norm"]),
            "happiness_score":          float(row["happiness_score_norm"]),
            "focus_score":              float(row["focus_score_norm"]),
            "productivity_score":       float(row["productivity_score_norm"]),
            "digital_dependence_score": float(row["digital_dependence_norm"]),
            "sleep_quality":            float(row["sleep_quality_norm"]),
            # Addiction / risk
            "addiction_severity":       int(row["addiction_severity"]),
            "high_risk_flag":           bool(row["high_risk_flag"]),
            # Device
            "device_type":              ["Android","iPhone","Laptop","Tablet","Unknown"][int(row["device_type_encoded"])],
            # Derived
            "usage_pattern":            usage_pattern,
            "screen_time_category":     screen_cat,
            # Extra
            "physical_activity_days":   int(row["physical_activity_days"]),
            "phone_unlocks":            int(row["phone_unlocks"]),
            "app_opens_per_day":        int(row["app_opens_per_day"]),
            "weekend_screen_time":      float(row["weekend_screen_time"]),
            "battery_drain_percent":    float(row["battery_drain_percent"]),
            "region":                   str(row["region"]),
            "income_level":             str(row["income_level"]),
            "education_level":          str(row["education_level"]),
            "daily_role":               str(row["daily_role"]),
        }

    print(f"   Usage patterns: { {p: sum(1 for v in profiles.values() if v['usage_pattern']==p) for p in ['social_media_heavy','gaming_heavy','productive','balanced']} }")
    return profiles

# ── Training helpers ──────────────────────────────────────────────────────────

def train_agents(env, env_name, num_episodes_q, num_episodes_dqn, log_every=100):
    state_size  = env.state_size
    action_size = env.action_space.n
    print(f"\n  State size={state_size}, Action size={action_size}")

    # Q-Learning
    print(f"  Training Q-Learning ({num_episodes_q} episodes)...")
    q_agent = QLearningAgent(
        state_space_size=2000,
        action_space_size=action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
    )
    q_rewards = []
    for ep in range(num_episodes_q):
        r, _ = q_agent.train_episode(env, max_steps=24*7)
        q_rewards.append(r)
        if (ep + 1) % log_every == 0:
            print(f"    Q  ep {ep+1:>4}/{num_episodes_q} | avg(last {log_every}): {np.mean(q_rewards[-log_every:]):+.2f}")

    # DQN
    print(f"  Training DQN ({num_episodes_dqn} episodes)...")
    dqn_agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        memory_size=10000,
        batch_size=32,
        target_update_freq=500,
    )
    dqn_rewards = []
    for ep in range(num_episodes_dqn):
        r, _ = dqn_agent.train_episode(env, max_steps=24*7)
        dqn_rewards.append(r)
        if (ep + 1) % log_every == 0:
            print(f"    DQN ep {ep+1:>4}/{num_episodes_dqn} | avg(last {log_every}): {np.mean(dqn_rewards[-log_every:]):+.2f}")

    return q_agent, dqn_agent, q_rewards, dqn_rewards

def evaluate_agent(agent, env, n=100):
    rewards = []
    for _ in range(n):
        state = env.reset()
        total, done, steps = 0, False, 0
        while not done and steps < 24*7:
            action = agent.choose_action(state, training=False)
            state, r, done, _ = env.step(action)
            total += r
            steps += 1
        rewards.append(total)
    return {
        "mean":  round(float(np.mean(rewards)), 3),
        "std":   round(float(np.std(rewards)),  3),
        "min":   round(float(np.min(rewards)),  3),
        "max":   round(float(np.max(rewards)),  3),
        "final": round(float(np.mean(rewards[-50:])), 3),
    }

def save_plot(q_rewards, dqn_rewards, env_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{env_name} — Training on final_unified_dataset.csv", fontsize=13, fontweight='bold')

    # Raw + smoothed
    w = 30
    for ax, rewards, label, color in [
        (axes[0], q_rewards,   "Q-Learning", "#22d3ee"),
        (axes[0], dqn_rewards, "DQN",        "#6366f1"),
    ]:
        ax.plot(rewards, alpha=0.2, color=color)
        if len(rewards) >= w:
            sm = np.convolve(rewards, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, len(rewards)), sm, color=color, linewidth=2, label=label)
        else:
            ax.plot(rewards, color=color, linewidth=2, label=label)

    axes[0].set_title("Episode Rewards (smoothed)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Comparison bar
    names  = ["Q-Learning", "DQN"]
    means  = [np.mean(q_rewards[-100:]) if len(q_rewards)>=100 else np.mean(q_rewards),
              np.mean(dqn_rewards[-100:]) if len(dqn_rewards)>=100 else np.mean(dqn_rewards)]
    colors = ["#22d3ee", "#6366f1"]
    bars   = axes[1].bar(names, means, color=colors, width=0.4)
    for bar, val in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1].set_title("Final Mean Reward (last 100 eps)")
    axes[1].set_ylabel("Mean Reward")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = env_name.lower().replace(" ", "_")
    path  = os.path.join(PLOTS_DIR, f"dataset_{fname}_training.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {path}")

# ── Main training loop ────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  RL TRAINING ON final_unified_dataset.csv")
    print("=" * 60)

    profiles = load_dataset_profiles()

    # Use a representative sample for speed (all 11k users makes env init slow)
    # Sample 500 users covering all sources proportionally
    import random
    random.seed(42)
    np.random.seed(42)

    sources = {}
    for uid, p in profiles.items():
        src = p["data_source"]
        sources.setdefault(src, []).append(uid)

    sampled_ids = []
    for src, ids in sources.items():
        n = min(500, len(ids))
        sampled_ids.extend(random.sample(ids, n))

    sample_profiles = {uid: profiles[uid] for uid in sampled_ids}
    print(f"\n  Using {len(sample_profiles)} sampled profiles for training")
    for src in sources:
        cnt = sum(1 for p in sample_profiles.values() if p["data_source"] == src)
        print(f"    {src}: {cnt}")

    summary = {}

    # ── 1. App Usage Control ──────────────────────────────────────────────────
    print("\n" + "─"*50)
    print("📱 App Usage Control")
    print("─"*50)
    env = AppUsageControlEnvironment(sample_profiles)
    q_agent, dqn_agent, q_r, dqn_r = train_agents(env, "app_usage", 500, 500, log_every=100)

    q_eval   = evaluate_agent(q_agent,   env, 50)
    dqn_eval = evaluate_agent(dqn_agent, env, 50)
    print(f"  Q-Learning eval: {q_eval}")
    print(f"  DQN eval:        {dqn_eval}")

    dqn_agent.save_model(os.path.join(MODELS_DIR, "dataset_app_usage_dqn_model.pth"))
    np.save(os.path.join(MODELS_DIR, "dataset_app_usage_q_rewards.npy"),   np.array(q_r))
    np.save(os.path.join(MODELS_DIR, "dataset_app_usage_dqn_rewards.npy"), np.array(dqn_r))
    save_plot(q_r, dqn_r, "App Usage Control")
    summary["app_usage"] = {"q": q_eval, "dqn": dqn_eval,
                             "q_episodes": len(q_r), "dqn_episodes": len(dqn_r)}

    # ── 2. Notification Scheduling ────────────────────────────────────────────
    print("\n" + "─"*50)
    print("🔔 Notification Scheduling")
    print("─"*50)
    env = NotificationSchedulingEnvironment(sample_profiles)
    q_agent, dqn_agent, q_r, dqn_r = train_agents(env, "notification", 500, 500, log_every=100)

    q_eval   = evaluate_agent(q_agent,   env, 50)
    dqn_eval = evaluate_agent(dqn_agent, env, 50)
    print(f"  Q-Learning eval: {q_eval}")
    print(f"  DQN eval:        {dqn_eval}")

    dqn_agent.save_model(os.path.join(MODELS_DIR, "dataset_notification_dqn_model.pth"))
    np.save(os.path.join(MODELS_DIR, "dataset_notification_q_rewards.npy"),   np.array(q_r))
    np.save(os.path.join(MODELS_DIR, "dataset_notification_dqn_rewards.npy"), np.array(dqn_r))
    save_plot(q_r, dqn_r, "Notification Scheduling")
    summary["notification"] = {"q": q_eval, "dqn": dqn_eval,
                                "q_episodes": len(q_r), "dqn_episodes": len(dqn_r)}

    # ── 3. Wellbeing Optimization ─────────────────────────────────────────────
    print("\n" + "─"*50)
    print("🧘 Wellbeing Optimization")
    print("─"*50)
    env = DigitalWellbeingOptimizationEnvironment(sample_profiles)
    q_agent, dqn_agent, q_r, dqn_r = train_agents(env, "wellbeing", 300, 300, log_every=50)

    q_eval   = evaluate_agent(q_agent,   env, 50)
    dqn_eval = evaluate_agent(dqn_agent, env, 50)
    print(f"  Q-Learning eval: {q_eval}")
    print(f"  DQN eval:        {dqn_eval}")

    dqn_agent.save_model(os.path.join(MODELS_DIR, "dataset_wellbeing_dqn_model.pth"))
    np.save(os.path.join(MODELS_DIR, "dataset_wellbeing_q_rewards.npy"),   np.array(q_r))
    np.save(os.path.join(MODELS_DIR, "dataset_wellbeing_dqn_rewards.npy"), np.array(dqn_r))
    save_plot(q_r, dqn_r, "Wellbeing Optimization")
    summary["wellbeing"] = {"q": q_eval, "dqn": dqn_eval,
                             "q_episodes": len(q_r), "dqn_episodes": len(dqn_r)}

    # ── Save summary ──────────────────────────────────────────────────────────
    summary["meta"] = {
        "dataset":       "final_unified_dataset.csv",
        "total_users":   len(profiles),
        "sampled_users": len(sample_profiles),
        "duration_min":  round((time.time() - t0) / 60, 1),
        "trained_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(REPORTS_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Duration : {summary['meta']['duration_min']} min")
    print(f"  Models   → results/models/dataset_*.pth")
    print(f"  Rewards  → results/models/dataset_*.npy")
    print(f"  Plots    → results/plots/dataset_*_training.png")
    print(f"  Summary  → results/reports/training_summary.json")
    print()
    for env_name, res in summary.items():
        if env_name == "meta":
            continue
        print(f"  {env_name}:")
        print(f"    Q-Learning  mean={res['q']['mean']:+.3f}  max={res['q']['max']:+.3f}")
        print(f"    DQN         mean={res['dqn']['mean']:+.3f}  max={res['dqn']['max']:+.3f}")

if __name__ == "__main__":
    main()
