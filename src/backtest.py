import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from stable_baselines3 import PPO
from finrl.meta.preprocessor.preprocessors import data_split
from env.risk_aware_env import RiskAwareEnv
import config


class Backtester:
    def __init__(self):
        self.processed_data_path = config.DATA_SAVE_PATH
        self.models_dir = config.TRAINED_MODEL_DIR
        self.results_dir = "./results"
        os.makedirs(self.results_dir, exist_ok=True)

    def _build_env_kwargs(self, df):
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

    def _compute_metrics(self, values):
        values = np.array(values, dtype=float)
        if values.shape[0] < 2:
            return {
                "profile": "",
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
            "profile": "",
            "final_value": final_value,
            "total_return_pct": total_return_pct,
            "sharpe": sharpe,
            "volatility_pct": volatility_pct,
            "max_drawdown_pct": max_drawdown_pct,
        }

    def _save_metrics(self, rows):
        if not rows:
            return
        df_metrics = pd.DataFrame(rows)
        cols = [
            "profile",
            "final_value",
            "total_return_pct",
            "sharpe",
            "volatility_pct",
            "max_drawdown_pct",
        ]
        df_metrics = df_metrics[cols]
        path = os.path.join(self.results_dir, "backtest_metrics.csv")
        df_metrics.to_csv(path, index=False)
        print(f"Metrik dosyası kaydedildi: {path}")

    def _save_series(self, profile_name, days, values, raw_rewards, adj_rewards, vols, penalties):
        df_series = pd.DataFrame(
            {
                "day": days,
                "portfolio_value": values,
                "raw_reward": raw_rewards,
                "risk_adjusted_reward": adj_rewards,
                "volatility_metric": vols,
                "risk_penalty": penalties,
            }
        )
        path = os.path.join(self.results_dir, f"series_{profile_name}.csv")
        df_series.to_csv(path, index=False)
        print(f"Zaman serisi kaydedildi: {path}")

    def _get_benchmark_data(self):
        try:
            df = yf.download("^DJI", start=config.TEST_START_DATE, end=config.TEST_END_DATE)
        except Exception as e:
            print(f"Benchmark indirilemedi: {e}")
            return pd.DataFrame()
        if df is None or df.empty:
            print("Benchmark verisi boş.")
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Adj Close" in df.columns:
            close = df["Adj Close"]
        elif "Close" in df.columns:
            close = df["Close"]
        else:
            close = df[df.columns[0]]
        out = pd.DataFrame({"close": close.values})
        return out

    def _plot_results(self, portfolio_values):
        if not portfolio_values:
            print("Portföy verisi yok, grafik çizilemedi.")
            return
        plt.figure(figsize=(12, 6))
        min_len = min(len(v) for v in portfolio_values.values())
        for profile, values in portfolio_values.items():
            vals = np.array(values[:min_len], dtype=float)
            initial = vals[0]
            returns = (vals - initial) / initial * 100.0
            plt.plot(returns, label=f"AI Agent ({profile})", linewidth=2)
        df_bench = self._get_benchmark_data()
        if not df_bench.empty:
            bench_vals = df_bench["close"].values
            if bench_vals.shape[0] >= min_len:
                bench_vals = bench_vals[:min_len]
            else:
                min_len = bench_vals.shape[0]
                for profile in list(portfolio_values.keys()):
                    portfolio_values[profile] = portfolio_values[profile][:min_len]
                bench_vals = bench_vals[:min_len]
            initial_bench = bench_vals[0]
            bench_returns = (bench_vals - initial_bench) / initial_bench * 100.0
            plt.plot(bench_returns, label="Benchmark (DJIA)", linestyle="--", linewidth=2)
        plt.title(f"Risk Duyarlı Ajan Performansı ({config.TEST_START_DATE} - {config.TEST_END_DATE})")
        plt.xlabel("İşlem Günleri")
        plt.ylabel("Kümülatif Getiri (%)")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(self.results_dir, "backtest_comparison.png")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Grafik kaydedildi: {save_path}")

    def _compute_portfolio_value_from_state(self, state, stock_dim):
        cash = state[0]
        shares = state[1 : 1 + stock_dim]
        prices = state[1 + stock_dim : 1 + 2 * stock_dim]
        value = float(cash + np.sum(np.array(shares) * np.array(prices)))
        return value

    def run_backtest(self):
        print("--- Backtest Başlıyor ---")
        df = pd.read_csv(self.processed_data_path)
        test_data = data_split(df, config.TEST_START_DATE, config.TEST_END_DATE)
        env_kwargs = self._build_env_kwargs(test_data)
        portfolio_values = {}
        metrics_rows = []
        for profile_name in config.RISK_PROFILES.keys():
            print(f"Test ediliyor: {profile_name}")
            env = RiskAwareEnv(df=test_data, risk_profile=profile_name, **env_kwargs)
            model_path = os.path.join(self.models_dir, f"ppo_{profile_name}")
            if not os.path.exists(model_path + ".zip"):
                print(f"Model bulunamadı: {model_path}.zip, atlanıyor.")
                continue
            model = PPO.load(model_path)
            obs, _ = env.reset()
            done = False
            truncated = False
            values = []
            days = []
            raw_rewards = []
            adj_rewards = []
            vols = []
            penalties = []
            v0 = self._compute_portfolio_value_from_state(obs, env.stock_dim)
            values.append(v0)
            days.append(0)
            raw_rewards.append(0.0)
            adj_rewards.append(0.0)
            vols.append(0.0)
            penalties.append(0.0)
            step_idx = 1
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                v = self._compute_portfolio_value_from_state(obs, env.stock_dim)
                values.append(v)
                days.append(step_idx)
                raw_rewards.append(float(info.get("raw_reward", reward)))
                adj_rewards.append(float(info.get("risk_adjusted_reward", reward)))
                vols.append(float(info.get("volatility_metric", 0.0)))
                penalties.append(float(info.get("risk_penalty", 0.0)))
                step_idx += 1
            portfolio_values[profile_name] = values
            m = self._compute_metrics(values)
            m["profile"] = profile_name
            metrics_rows.append(m)
            self._save_series(profile_name, days, values, raw_rewards, adj_rewards, vols, penalties)
        self._save_metrics(metrics_rows)
        self._plot_results(portfolio_values)
