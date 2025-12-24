import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "./results"


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_backtest_metrics():
    path = os.path.join(RESULTS_DIR, "backtest_metrics.csv")
    if not os.path.exists(path):
        print("backtest_metrics.csv bulunamadı, atlanıyor.")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("backtest_metrics.csv boş, atlanıyor.")
        return
    profiles = df["profile"].astype(str).tolist()
    x = np.arange(len(profiles))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, df["total_return_pct"], width, label="Total Return (%)")
    ax.bar(x, df["sharpe"], width, label="Sharpe")
    ax.bar(x + width, df["max_drawdown_pct"], width, label="Max Drawdown (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles)
    ax.set_title("Backtest Metrics by Risk Profile")
    ax.set_xlabel("Risk Profile")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "analysis_backtest_metrics.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Backtest metrik grafiği kaydedildi: {save_path}")


def plot_series_files():
    pattern = os.path.join(RESULTS_DIR, "series_*.csv")
    files = glob.glob(pattern)
    if not files:
        print("series_*.csv bulunamadı, atlanıyor.")
        return
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        name = os.path.splitext(os.path.basename(path))[0].replace("series_", "")
        if "day" not in df.columns or "portfolio_value" not in df.columns:
            continue
        days = df["day"].values
        values = df["portfolio_value"].values
        raw_rewards = df["raw_reward"].values if "raw_reward" in df.columns else None
        adj_rewards = df["risk_adjusted_reward"].values if "risk_adjusted_reward" in df.columns else None
        vols = df["volatility_metric"].values if "volatility_metric" in df.columns else None

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(days, values, linewidth=1.5)
        axes[0].set_ylabel("Portfolio Value")
        axes[0].set_title(f"Time Series – {name}")

        if raw_rewards is not None and adj_rewards is not None:
            axes[1].plot(days, raw_rewards, linewidth=1.0, label="Raw Reward")
            axes[1].plot(days, adj_rewards, linewidth=1.0, label="Risk-Adjusted")
            axes[1].set_ylabel("Reward")
            axes[1].legend()
        else:
            axes[1].axis("off")

        if vols is not None:
            axes[2].plot(days, vols, linewidth=1.0)
            axes[2].set_ylabel("Volatility Metric")
        else:
            axes[2].axis("off")

        axes[2].set_xlabel("Step")
        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f"analysis_series_{name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Zaman serisi grafiği kaydedildi: {save_path}")


def plot_ablation_metrics():
    path = os.path.join(RESULTS_DIR, "ablation_metrics.csv")
    if not os.path.exists(path):
        print("ablation_metrics.csv bulunamadı, atlanıyor.")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("ablation_metrics.csv boş, atlanıyor.")
        return
    experiments = df["experiment"].astype(str).tolist()
    x = np.arange(len(experiments))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, df["sharpe"], width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15)
    ax.set_title("Ablation Study – Sharpe by Experiment")
    ax.set_ylabel("Sharpe")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "analysis_ablation_sharpe.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Ablation Sharpe grafiği kaydedildi: {save_path}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["volatility_pct"], df["total_return_pct"])
    for i, label in enumerate(experiments):
        ax.annotate(label, (df["volatility_pct"][i], df["total_return_pct"][i]))
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Ablation Study – Risk/Return Diagram")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "analysis_ablation_risk_return.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Ablation risk-getiri grafiği kaydedildi: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        type=str,
        default="all",
        choices=["all", "backtest", "series", "ablation"],
    )
    args = parser.parse_args()
    ensure_results_dir()
    if args.section in ("all", "backtest"):
        plot_backtest_metrics()
    if args.section in ("all", "series"):
        plot_series_files()
    if args.section in ("all", "ablation"):
        plot_ablation_metrics()


if __name__ == "__main__":
    main()
