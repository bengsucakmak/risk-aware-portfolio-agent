import argparse
import os
import pandas as pd
from finrl.meta.preprocessor.preprocessors import data_split
from env.risk_aware_env import RiskAwareEnv
from src.data_processor import DataProcessor
from src.train import AgentTrainer
from src.backtest import Backtester
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


def run_data_prep():
    print("--- Mod: Veri Hazırlama ---")
    os.makedirs("data", exist_ok=True)
    processor = DataProcessor()
    processor.run()


def run_env_test():
    print("--- Mod: Ortam Testi ---")
    if not os.path.exists(config.DATA_SAVE_PATH):
        print("Hata: Veri bulunamadı. Önce --mode data çalıştır.")
        return
    df = pd.read_csv(config.DATA_SAVE_PATH)
    test_data = data_split(df, config.TEST_START_DATE, config.TEST_END_DATE)
    env_kwargs = build_env_kwargs(test_data)
    env = RiskAwareEnv(df=test_data, risk_profile="balanced", **env_kwargs)
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Test adımı tamamlandı. Ödül: {reward}")


def run_training(args):
    print("--- Mod: Ajan Eğitimi (PPO) ---")
    if not os.path.exists(config.DATA_SAVE_PATH):
        print("Hata: Veri bulunamadı. Önce --mode data çalıştır.")
        return
    df = pd.read_csv(config.DATA_SAVE_PATH)
    print("> Veri yüklendi:", df.shape)
    profiles = list(config.RISK_PROFILES.keys())
    if args.profile != "all":
        if args.profile not in profiles:
            print(f"Geçersiz profil: {args.profile}")
            return
        profiles = [args.profile]
    original_timesteps = config.PPO_TOTAL_TIMESTEPS
    if args.timesteps is not None:
        config.PPO_TOTAL_TIMESTEPS = int(args.timesteps)
    trainer = AgentTrainer(df)
    for p in profiles:
        trainer.train_agent(p)
    config.PPO_TOTAL_TIMESTEPS = original_timesteps


def run_backtest():
    print("--- Mod: Backtest ---")
    backtester = Backtester()
    backtester.run_backtest()

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["data", "test_env", "train", "backtest"],
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="all",
        choices=["all", "conservative", "balanced", "aggressive"],
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    if args.mode == "data":
        run_data_prep()
    elif args.mode == "test_env":
        run_env_test()
    elif args.mode == "train":
        run_training(args)
    elif args.mode == "backtest":
        run_backtest()


if __name__ == "__main__":
    main()
