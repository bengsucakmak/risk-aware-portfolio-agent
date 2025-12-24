import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from finrl.meta.preprocessor.preprocessors import data_split
from env.risk_aware_env import RiskAwareEnv
import config


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProgressCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 2000 == 0:
            print(f"[Training Update] Steps: {self.n_calls}")
        return True


class AgentTrainer:
    def __init__(self, df):
        self.df = df
        self.models_dir = config.TRAINED_MODEL_DIR
        self.train_data = data_split(
            self.df, config.TRAIN_START_DATE, config.TRAIN_END_DATE
        )

    def _build_env_kwargs(self):
        stock_dimension = len(self.train_data["tic"].unique())
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

    def _make_env(self, risk_profile: str, env_kwargs):
        def _init():
            return RiskAwareEnv(
                df=self.train_data,
                risk_profile=risk_profile,
                **env_kwargs,
            )

        return _init

    def train_agent(self, risk_profile: str):
        print(f"\n=== TRAINING PPO AGENT: {risk_profile.upper()} ===")

        set_global_seed(config.SEED)

        env_kwargs = self._build_env_kwargs()
        env_train = DummyVecEnv([self._make_env(risk_profile, env_kwargs)])

        os.makedirs(os.path.join(self.models_dir, "checkpoints"), exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=config.CHECKPOINT_FREQ,
            save_path=os.path.join(self.models_dir, "checkpoints"),
            name_prefix=f"ppo_{risk_profile}",
        )

        progress_callback = ProgressCallback()

        model = PPO(
            "MlpPolicy",
            env_train,
            learning_rate=config.PPO_LR,
            n_steps=config.PPO_N_STEPS,
            batch_size=config.PPO_BATCH_SIZE,
            ent_coef=config.PPO_ENT_COEF,
            gamma=config.PPO_GAMMA,
            verbose=1,
            seed=config.SEED,
        )

        model.learn(
            total_timesteps=config.PPO_TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, progress_callback],
        )

        os.makedirs(self.models_dir, exist_ok=True)
        save_path = os.path.join(self.models_dir, f"ppo_{risk_profile}")
        model.save(save_path)
        print(f"> Model saved: {save_path}.zip\n")
