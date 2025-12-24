import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import config

RISK_PENALTY_FACTORS = {
    "conservative": config.RISK_PROFILES["conservative"],
    "balanced": config.RISK_PROFILES["balanced"],
    "aggressive": config.RISK_PROFILES["aggressive"],
}


class RiskAwareEnv(StockTradingEnv):
    def __init__(self, risk_profile: str = "balanced", **kwargs):
        self.risk_profile = risk_profile
        self.risk_penalty_factor = RISK_PENALTY_FACTORS.get(
            risk_profile, config.RISK_PROFILES["balanced"]
        )
        super().__init__(**kwargs)

    def _compute_volatility_metric(self) -> float:
        start_idx = self.day * self.stock_dim
        end_idx = start_idx + self.stock_dim
        df_day = self.df.iloc[start_idx:end_idx]

        if "atr" in df_day.columns:
            vol = df_day["atr"].mean()
        else:
            vol = (df_day["high"] - df_day["low"]).mean()

        if np.isnan(vol):
            vol = 0.0

        return float(vol)

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)

        vol_metric = self._compute_volatility_metric()
        risk_penalty = self.risk_penalty_factor * vol_metric
        risk_adjusted_reward = reward - risk_penalty

        if self.risk_profile == "conservative" and reward < 0:
            risk_adjusted_reward *= 1.5

        self.reward = risk_adjusted_reward

        if info is None:
            info = {}

        info.update(
            {
                "raw_reward": float(reward),
                "risk_adjusted_reward": float(risk_adjusted_reward),
                "volatility_metric": float(vol_metric),
                "risk_penalty": float(risk_penalty),
                "risk_profile": self.risk_profile,
            }
        )

        return obs, self.reward, terminated, truncated, info
