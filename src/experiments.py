import os
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from finrl.meta.preprocessor.preprocessors import data_split
from env.risk_aware_env import RiskAwareEnv
from src.data_processor import DataProcessor
from src.train import AgentTrainer
import config


def build_env_kwargs(df):
    stock_dimension = len(df["tic"].unique())
    state_space = 1 + 2 * stock_dimension + len(config.INDICATORS) * stock_dimension
    num_stock_shares = [0] * stock_dimension
    buy_cost_list = [0.001] * stock_dimension
    sell_cost_list = [0.001] * stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }
    return env_kwargs


def compute_metrics(values):
    values = np.array(values, dtype=float)
    if values.shape[0] < 2:
        return {
            "final_value": float(values[-1]) if values.shape[0] > 0 else 0.0,
            "total_return_pct": 0.0,
            "sharpe": 0.0,
            "volatility_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }
    returns = np.diff(values) / values[:-1]
    avg_ret = float(returns.mean())
    vol = float(returns.std() + 1e-8)
    sharpe = float((avg_ret / vol) * np.sqrt(252.0))
    cummax = np.maximum.accumulate(values)
    drawdowns = (cummax - values) / cummax
    max_drawdown = float(drawdowns.max())
    final_value = float(values[-1])
    total_return_pct = float((values[-1] - values[0]) / values[0] * 100.0)
    volatility_pct = float(vol * np.sqrt(252.0) * 100.0)
    max_drawdown_pct = float(max_drawdown * 100.0)
    return {
        "final_value": final_value,
        "total_return_pct": total_return_pct,
        "sharpe": sharpe,
        "volatility_pct": volatility_pct,
        "max_drawdown_pct": max_drawdown_pct,
    }


def compute_portfolio_value_from_state(state, stock_dim):
    cash = state[0]
    shares = state[1 : 1 + stock_dim]
    prices = state[1 + stock_dim : 1 + 2 * stock_dim]
    value = float(cash + np.sum(np.array(shares) * np.array(prices)))
    return value


def run_single_backtest(df, model_dir):
    test_data = data_split(df, config.TEST_START_DATE, config.TEST_END_DATE)
    env_kwargs = build_env_kwargs(test_data)
    env = RiskAwareEnv(df=test_data, risk_profile="balanced", **env_kwargs)
    model_path = os.path.join(model_dir, "ppo_balanced")
    if not os.path.exists(model_path + ".zip"):
        return None
    model = PPO.load(model_path)
    obs, _ = env.reset()
    done = False
    truncated = False
    values = []
    v0 = compute_portfolio_value_from_state(obs, env.stock_dim)
    values.append(v0)
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        v = compute_portfolio_value_from_state(obs, env.stock_dim)
        values.append(v)
    m = compute_metrics(values)
    return m


def get_indicator_sets():
    full = list(config.INDICATORS)
    risk_inds = {"atr", "boll_ub", "boll_lb"}
    momentum_inds = {"rsi_30", "cci_30", "dx_30", "macd"}
    no_risk = [i for i in full if i not in risk_inds]
    risk_only = [i for i in full if i in risk_inds]
    no_momentum = [i for i in full if i not in momentum_inds]
    sets = {
        "full": full,
        "no_risk": no_risk,
        "risk_only": risk_only,
        "no_momentum": no_momentum,
    }
    return sets


def run_ablation(timesteps_override=None):
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    original_indicators = list(config.INDICATORS)
    original_data_path = config.DATA_SAVE_PATH
    original_model_dir = config.TRAINED_MODEL_DIR
    original_timesteps = getattr(config, "PPO_TOTAL_TIMESTEPS", None)
    indicator_sets = get_indicator_sets()
    rows = []
    for name, indicators in indicator_sets.items():
        print(f"=== Ablation: {name} ===")
        config.INDICATORS = indicators
        config.DATA_SAVE_PATH = f"./data/processed_{name}.csv"
        config.TRAINED_MODEL_DIR = f"./models_ablation/{name}"
        if timesteps_override is not None and original_timesteps is not None:
            config.PPO_TOTAL_TIMESTEPS = int(timesteps_override)
        processor = DataProcessor()
        df = processor.run()
        trainer = AgentTrainer(df)
        trainer.train_agent("balanced")
        m = run_single_backtest(df, config.TRAINED_MODEL_DIR)
        if m is None:
            continue
        m["experiment"] = name
        rows.append(m)
    config.INDICATORS = original_indicators
    config.DATA_SAVE_PATH = original_data_path
    config.TRAINED_MODEL_DIR = original_model_dir
    if original_timesteps is not None:
        config.PPO_TOTAL_TIMESTEPS = original_timesteps
    if rows:
        df_metrics = pd.DataFrame(rows)
        cols = [
            "experiment",
            "final_value",
            "total_return_pct",
            "sharpe",
            "volatility_pct",
            "max_drawdown_pct",
        ]
        df_metrics = df_metrics[cols]
        out_path = "./results/ablation_metrics.csv"
        df_metrics.to_csv(out_path, index=False)
        print(f"Ablation metrikleri kaydedildi: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    run_ablation(timesteps_override=args.timesteps)


if __name__ == "__main__":
    main()
