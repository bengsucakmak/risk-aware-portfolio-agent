import datetime

TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2023-01-01"

TEST_START_DATE = "2023-01-02"
TEST_END_DATE = "2024-01-01"

# DÜZELTME: 'mach_30' çıkarıldı.
# 'mac' (Moving Average Convergence Divergence) standart olarak eklendi.
INDICATORS = [
    "rsi_30",        # Relative Strength Index
    "cci_30",        # Commodity Channel Index
    "dx_30",         # Directional Movement Index
    "atr",           # Average True Range (RISK İÇİN KRİTİK)
    "boll_ub",       # Bollinger Upper Band (RISK İÇİN KRİTİK)
    "boll_lb",       # Bollinger Lower Band (RISK İÇİN KRİTİK)
    "close_30_sma",  # Simple Moving Average
    "macd"           # Standart MACD (Suffix olmadan)
]

TICKER_LIST = [
    "AAPL", "MSFT", "JPM", "V", "RTX", "PG", "GS", "NKE", "DIS", "AXP",
    "HD", "INTC", "WMT", "IBM", "MRK", "UNH", "KO", "CAT", "TRV", "JNJ",
    "CVX", "MCD", "VZ", "CSCO", "XOM", "BA", "MMM", "PFE"
]

RISK_PROFILES = {
    "conservative": 0.9,
    "balanced": 0.5,
    "aggressive": 0.1
}

DATA_SAVE_PATH = "./data/processed_data.csv"
TRAINED_MODEL_DIR = "./models"

SEED = 42

PPO_LR = 0.00025
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_ENT_COEF = 0.0
PPO_GAMMA = 0.99
PPO_TOTAL_TIMESTEPS = 10000

CHECKPOINT_FREQ = 5000
