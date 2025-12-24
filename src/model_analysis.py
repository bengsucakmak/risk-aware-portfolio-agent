import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from finrl.meta.preprocessor.preprocessors import data_split
from env.risk_aware_env import RiskAwareEnv
import config


RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_env_kwargs(df):
    stock_dim = len(df["tic"].unique())
    state_space = 1 + 2 * stock_dim + len(config.INDICATORS) * stock_dim
    num_stock = [0] * stock_dim
    buy_cost = [0.001] * stock_dim
    sell_cost = [0.001] * stock_dim
    return {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock,
        "buy_cost_pct": buy_cost,
        "sell_cost_pct": sell_cost,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": config.INDICATORS,
        "action_space": stock_dim,
        "reward_scaling": 1e-4,
    }


def analyze_weights(model, name):
    params = []
    for p in model.policy.parameters():
        arr = p.detach().cpu().numpy().flatten()
        params.append(arr)
    w = np.concatenate(params)
    plt.figure(figsize=(7, 4))
    plt.hist(w, bins=60)
    plt.title(f"Policy Weight Distribution – {name}")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"model_weights_{name}.png")
    plt.savefig(out)
    plt.close()
    print("Kaydedildi:", out)


def analyze_policy_distribution(model, env, name):
    actions = []
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=float)
    for _ in range(400):
        action, _ = model.predict(obs, deterministic=False)
        actions.append(action)
        obs, _, done, truncated, _ = env.step(action)
        obs = np.asarray(obs, dtype=float)
        if done or truncated:
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=float)
    actions = np.array(actions)
    mean_actions = actions.mean(axis=0)
    plt.figure(figsize=(9, 4))
    plt.plot(mean_actions)
    plt.title(f"Policy Action Mean per Asset – {name}")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"model_policy_dist_{name}.png")
    plt.savefig(out)
    plt.close()
    print("Kaydedildi:", out)


def action_sensitivity(model, env, name):
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=float)
    prices_index = 1 + env.stock_dim
    prices_end = 1 + 2 * env.stock_dim
    base_action, _ = model.predict(obs, deterministic=True)
    sensitivity = []
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        new_obs = obs.copy()
        new_obs[prices_index:prices_end] = new_obs[prices_index:prices_end] * scale
        a, _ = model.predict(new_obs, deterministic=True)
        diff = np.abs(a - base_action).mean()
        sensitivity.append(diff)
    plt.figure(figsize=(7, 4))
    plt.plot([80, 90, 100, 110, 120], sensitivity, marker="o")
    plt.title(f"Action Sensitivity to Price Changes – {name}")
    plt.xlabel("Price Level (%)")
    plt.ylabel("Mean Action Difference")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"model_action_sensitivity_{name}.png")
    plt.savefig(out)
    plt.close()
    print("Kaydedildi:", out)


def analyze_model(profile):
    df = pd.read_csv(config.DATA_SAVE_PATH)
    test_data = data_split(df, config.TEST_START_DATE, config.TEST_END_DATE)
    env_kwargs = build_env_kwargs(test_data)
    env = RiskAwareEnv(df=test_data, risk_profile=profile, **env_kwargs)
    model_path = os.path.join(config.TRAINED_MODEL_DIR, f"ppo_{profile}")
    if not os.path.exists(model_path + ".zip"):
        print("Model bulunamadı:", model_path)
        return
    model = PPO.load(model_path)
    analyze_weights(model, profile)
    analyze_policy_distribution(model, env, profile)
    action_sensitivity(model, env, profile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=str,
        default="all",
        choices=["all", "conservative", "balanced", "aggressive"],
    )
    args = parser.parse_args()
    profiles = ["conservative", "balanced", "aggressive"] if args.profile == "all" else [args.profile]
    for p in profiles:
        analyze_model(p)


if __name__ == "__main__":
    main()
