# src/b2c_app.py
from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- Project imports (robust) ---
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import config  # noqa: E402

try:
    from env.risk_aware_env import RiskAwareEnv  # noqa: E402
except Exception:  # pragma: no cover
    from src.risk_aware_env import RiskAwareEnv  # type: ignore # noqa: E402

try:
    from src.data_processor import DataProcessor  # noqa: E402
except Exception:  # pragma: no cover
    from data_processor import DataProcessor  # type: ignore # noqa: E402

try:
    from src.train import AgentTrainer  # noqa: E402
except Exception:  # pragma: no cover
    from train import AgentTrainer  # type: ignore # noqa: E402

try:
    from src.backtest import Backtester  # noqa: E402
except Exception:  # pragma: no cover
    from backtest import Backtester  # type: ignore # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from finrl.meta.preprocessor.preprocessors import data_split  # noqa: E402


RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = Path(getattr(config, "TRAINED_MODEL_DIR", str(BASE_DIR / "models")))

# ----------------------------
# UI / Theme helpers
# ----------------------------
PROFILE_META = {
    "conservative": {"label_tr": "Konservatif", "emoji": "🟦", "accent": "rgba(59,130,246,0.30)"},
    "balanced": {"label_tr": "Dengeli", "emoji": "🟩", "accent": "rgba(34,197,94,0.30)"},
    "aggressive": {"label_tr": "Agresif", "emoji": "🟥", "accent": "rgba(239,68,68,0.30)"},
}

# (İyileştirme #2) Teknik terimleri sadeleştirme
LABELS = {
    "reward": "Günlük Skor",
    "risk_penalty": "Risk Cezası",
    "env_day": "Gün",
    "pv": "Portföy Değeri",
}

METRIC_HELP = {
    "total_return_pct": "Backtest dönemi boyunca kümülatif getiri (%)",
    "sharpe": "Risk başına getiri (yıllıklaştırılmış Sharpe)",
    "volatility_pct": "Yıllıklaştırılmış volatilite (%)",
    "max_drawdown_pct": "Maksimum geri çekilme (%)",
}

def _profile_label(profile: str) -> str:
    meta = PROFILE_META.get(profile, {})
    return f"{meta.get('emoji', '•')} {meta.get('label_tr', profile)}"

def _shorten(text: str, max_len: int = 120) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .rp-card{border-radius:16px;padding:14px 14px;border:1px solid rgba(148,163,184,0.35);background:rgba(148,163,184,0.07);}
        .rp-pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;letter-spacing:0.2px;}
        .rp-row{display:flex;align-items:center;justify-content:space-between;gap:10px;}
        .rp-title{font-size:16px;font-weight:650;margin:0;}
        .rp-sub{opacity:0.75;font-size:12px;margin:2px 0 0 0;}
        .rp-metric{font-size:26px;font-weight:750;margin:6px 0 0 0;}
        .rp-muted{opacity:0.7;font-size:12px;}
        .rp-rec{border-radius:16px;padding:14px;border:1px solid rgba(148,163,184,0.30);background:rgba(148,163,184,0.06);}
        .rp-rec h4{margin:0;font-size:16px;}
        .rp-banner{border-radius:18px;padding:14px 14px;border:1px solid rgba(148,163,184,0.35);background:linear-gradient(90deg, rgba(59,130,246,0.12), rgba(34,197,94,0.10), rgba(239,68,68,0.10));}
        .rp-banner h3{margin:0;font-size:16px;font-weight:750;}
        .rp-steps{margin-top:8px;display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
        .rp-step{border-radius:14px;padding:10px;border:1px solid rgba(148,163,184,0.25);background:rgba(148,163,184,0.06);}
        .rp-step b{display:block;margin-bottom:4px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

def metric_card(title: str, value: str, profile: str, help_text: str = "") -> None:
    accent = PROFILE_META.get(profile, {}).get("accent", "rgba(148,163,184,0.25)")
    st.markdown(
        f"""
        <div class="rp-card">
            <div class="rp-row">
                <div>
                    <p class="rp-title">{title}</p>
                    <p class="rp-sub">{help_text}</p>
                </div>
                <span class="rp-pill" style="background:{accent};">{_profile_label(profile)}</span>
            </div>
            <div class="rp-metric">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# (İyileştirme #1) Onboarding banner (3 adım)
def onboarding_banner(profile: str) -> None:
    st.markdown(
        f"""
        <div class="rp-banner">
            <h3>3 Adımda Kullanım • Seçili Profil: {_profile_label(profile)}</h3>
            <div class="rp-steps">
                <div class="rp-step"><b>1) Profil seç</b> Risk yaklaşımını belirle (Konservatif / Dengeli / Agresif).</div>
                <div class="rp-step"><b>2) Önerileri incele</b> Kartlardan BUY/SELL/HOLD önerilerini gör ve istersen sanal portföye ekle.</div>
                <div class="rp-step"><b>3) Mini sim ile test et</b> Step seç → o günün kartlarını gör → 1 gün veya 10 gün ilerlet.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# File loaders (cached)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df is None or df.empty:
        return None
    return df

def _find_profile_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["profile", "risk_profile", "strategy", "name"]:
        if c in df.columns:
            return c
    return None

def _find_metric(row: pd.Series, candidates: List[str]) -> Optional[float]:
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                return None
    return None

def load_backtest_metrics() -> Optional[pd.DataFrame]:
    return load_csv(RESULTS_DIR / "backtest_metrics.csv")

def load_series(profile: str) -> Optional[pd.DataFrame]:
    return load_csv(RESULTS_DIR / f"series_{profile}.csv")

# ----------------------------
# RL recommendation (one-step)
# ----------------------------
@dataclass
class Recommendation:
    ticker: str
    action: str           # BUY / SELL / HOLD
    strength: float       # abs(action value)
    reason: str
    risk_note: str
    amount: Optional[int] = None

def _build_env_kwargs(df: pd.DataFrame) -> Dict[str, Any]:
    stock_dim = len(df["tic"].unique())
    state_space = 1 + 2 * stock_dim + len(config.INDICATORS) * stock_dim
    return {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": config.INDICATORS,
        "action_space": stock_dim,
        "reward_scaling": 1e-4,
    }

def _explain_from_indicators(row: pd.Series) -> Tuple[str, str]:
    reasons = []
    risk = []

    if "rsi_30" in row.index:
        rsi = float(row["rsi_30"])
        if rsi < 30:
            reasons.append(f"RSI düşük ({rsi:.1f}) → olası aşırı satım.")
        elif rsi > 70:
            reasons.append(f"RSI yüksek ({rsi:.1f}) → olası aşırı alım.")
    if "macd" in row.index:
        macd = float(row["macd"])
        if macd > 0:
            reasons.append("MACD pozitif → momentum yukarı.")
        elif macd < 0:
            reasons.append("MACD negatif → momentum aşağı.")
    if "atr" in row.index:
        atr = float(row["atr"])
        risk.append(f"ATR (volatilite) {atr:.4f}")
    if "boll_ub" in row.index and "boll_lb" in row.index and "close" in row.index:
        close = float(row["close"])
        ub = float(row["boll_ub"])
        lb = float(row["boll_lb"])
        if close >= ub:
            risk.append("Fiyat üst Bollinger bandına yakın/üstünde.")
        if close <= lb:
            risk.append("Fiyat alt Bollinger bandına yakın/altında.")

    reason = " • ".join(reasons) if reasons else "Teknik indikatör kombinasyonu (özet)."
    risk_note = " • ".join(risk) if risk else "Risk notu: indikatör/volatilite seviyesine göre."
    return reason, risk_note

def get_rl_recommendations(profile: str, top_k: int = 6) -> List[Recommendation]:
    df = load_csv(Path(config.DATA_SAVE_PATH))
    if df is None:
        raise FileNotFoundError(f"Veri yok: {config.DATA_SAVE_PATH}")

    test_df = data_split(df, config.TEST_START_DATE, config.TEST_END_DATE)
    if test_df is None or test_df.empty:
        raise ValueError("Test split boş.")

    env_kwargs = _build_env_kwargs(test_df)
    env = RiskAwareEnv(df=test_df, risk_profile=profile, **env_kwargs)

    model_path = MODELS_DIR / f"ppo_{profile}"
    if not (model_path.with_suffix(".zip")).exists():
        raise FileNotFoundError(f"Model yok: {model_path}.zip")

    model = PPO.load(str(model_path))

    obs, _ = env.reset()
    action_vec, _ = model.predict(obs, deterministic=True)
    action_vec = np.asarray(action_vec, dtype=float).reshape(-1)

    tics = sorted(test_df["tic"].unique().tolist())
    if len(tics) != len(action_vec):
        raise ValueError(f"Aksiyon boyutu ({len(action_vec)}) != hisse sayısı ({len(tics)})")

    idx_sorted = np.argsort(np.abs(action_vec))[::-1]
    picks = idx_sorted[: min(top_k, len(idx_sorted))]

    recs: List[Recommendation] = []
    for i in picks:
        tic = tics[int(i)]
        a = float(action_vec[int(i)])
        if a > 0.05:
            act = "BUY"
        elif a < -0.05:
            act = "SELL"
        else:
            act = "HOLD"

        last_row = test_df[test_df["tic"] == tic].iloc[-1]
        reason, risk_note = _explain_from_indicators(last_row)

        recs.append(
            Recommendation(
                ticker=tic,
                action=act,
                strength=abs(a),
                reason=reason,
                risk_note=risk_note,
                amount=int(min(100, max(1, round(abs(a) * 100)))) if act != "HOLD" else None,
            )
        )
    return recs

def get_dummy_recommendations(profile: str) -> List[Recommendation]:
    rng = np.random.default_rng(42)
    sample = ["AAPL", "MSFT", "JPM", "V", "PG", "DIS"]
    actions = ["BUY", "SELL", "HOLD"]
    recs = []
    for tic in sample:
        act = rng.choice(actions, p=[0.45, 0.35, 0.20])
        strength = float(rng.uniform(0.1, 1.0))
        recs.append(
            Recommendation(
                ticker=tic,
                action=act,
                strength=strength,
                reason="Demo amaçlı örnek öneri (RL fallback).",
                risk_note=f"Profil: {_profile_label(profile)}",
                amount=int(round(strength * 10)) if act != "HOLD" else None,
            )
        )
    return recs

# ----------------------------
# Session state (virtual portfolio + mini sim)
# ----------------------------
def init_virtual_portfolio(profile: str, initial_capital: float) -> None:
    st.session_state.virtual_portfolio = {
        "profile": profile,
        "initial_capital": float(initial_capital),
        "actions": [],
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    }

def ensure_virtual_portfolio(profile: str, initial_capital: float) -> None:
    vp = st.session_state.get("virtual_portfolio")
    if vp is None:
        init_virtual_portfolio(profile, initial_capital)
        return
    vp["profile"] = profile
    vp["initial_capital"] = float(initial_capital)

def accept_recommendation(rec: Recommendation, source: str = "cards") -> None:
    vp = st.session_state.virtual_portfolio
    vp["actions"].append(
        {
            "ts": datetime.utcnow().isoformat(timespec="seconds"),
            "source": source,
            "ticker": rec.ticker,
            "action": rec.action,
            "amount": rec.amount,
            "strength": float(rec.strength),
            "reason": rec.reason,
            "risk_note": rec.risk_note,
        }
    )

# (İyileştirme #3) Sanal portföy özet verisi
def virtual_portfolio_summary() -> Dict[str, Any]:
    vp = st.session_state.get("virtual_portfolio", {})
    actions = vp.get("actions", []) or []
    df = pd.DataFrame(actions) if actions else pd.DataFrame(columns=["action", "ts"])
    counts = df["action"].value_counts().to_dict() if not df.empty else {}
    total = int(sum(counts.values())) if counts else 0
    last3 = df.sort_values("ts", ascending=False).head(3).to_dict("records") if not df.empty else []
    buy = int(counts.get("BUY", 0))
    sell = int(counts.get("SELL", 0))
    hold = int(counts.get("HOLD", 0))
    return {
        "total": total,
        "buy": buy,
        "sell": sell,
        "hold": hold,
        "last3": last3,
    }

def ensure_live_sim_state() -> None:
    if "live_sim" not in st.session_state:
        st.session_state.live_sim = {
            "env": None,
            "model": None,
            "profile": None,
            "history": [],
            "step": 0,
            "done": False,
            "obs": None,
        }

def portfolio_value_from_state(state: np.ndarray, stock_dim: int) -> float:
    cash = float(state[0])
    shares = np.asarray(state[1 : 1 + stock_dim], dtype=float)
    prices = np.asarray(state[1 + stock_dim : 1 + 2 * stock_dim], dtype=float)
    return float(cash + np.sum(shares * prices))

def init_live_simulation(profile: str) -> None:
    ensure_live_sim_state()

    df = load_csv(Path(config.DATA_SAVE_PATH))
    if df is None:
        raise FileNotFoundError(f"Veri yok: {config.DATA_SAVE_PATH}")

    test_df = data_split(df, config.TEST_START_DATE, config.TEST_END_DATE)
    env_kwargs = _build_env_kwargs(test_df)
    env = RiskAwareEnv(df=test_df, risk_profile=profile, **env_kwargs)

    model_path = MODELS_DIR / f"ppo_{profile}"
    if not (model_path.with_suffix(".zip")).exists():
        raise FileNotFoundError(f"Model yok: {model_path}.zip")
    model = PPO.load(str(model_path))

    st.session_state.live_sim.update(
        {
            "env": env,
            "model": model,
            "profile": profile,
            "history": [],
            "step": 0,
            "done": False,
            "obs": None,
        }
    )

def _cards_from_action_vec(env: Any, action_vec: np.ndarray, top_k: int = 6) -> List[Dict[str, Any]]:
    action_vec = np.asarray(action_vec, dtype=float).reshape(-1)
    tics = sorted(env.df["tic"].unique().tolist())
    idx_sorted = np.argsort(np.abs(action_vec))[::-1][: min(top_k, len(action_vec))]
    cards: List[Dict[str, Any]] = []

    for i in idx_sorted:
        a = float(action_vec[int(i)])
        if a > 0.05:
            act = "BUY"
        elif a < -0.05:
            act = "SELL"
        else:
            act = "HOLD"

        tic = tics[int(i)] if int(i) < len(tics) else f"TIC_{i}"
        last_row = env.df[env.df["tic"] == tic].iloc[-1]
        reason, risk_note = _explain_from_indicators(last_row)

        cards.append(
            {
                "ticker": tic,
                "action": act,
                "strength": abs(a),
                "amount": int(min(100, max(1, round(abs(a) * 100)))) if act != "HOLD" else None,
                "reason": reason,
                "risk_note": risk_note,
            }
        )
    return cards

def step_live_sim() -> None:
    sim = st.session_state.live_sim
    env = sim["env"]
    model = sim["model"]
    if env is None or model is None:
        raise RuntimeError("Simülasyon başlatılmamış.")

    if sim["done"]:
        return

    obs = sim.get("obs")
    if obs is None:
        obs, _ = env.reset()
        sim["obs"] = obs

    action_vec, _ = model.predict(obs, deterministic=True)
    next_obs, reward, terminated, truncated, info = env.step(action_vec)
    sim["obs"] = next_obs

    pv = portfolio_value_from_state(np.asarray(next_obs, dtype=float), env.stock_dim)
    risk_penalty = float(info.get("risk_penalty", 0.0)) if isinstance(info, dict) else 0.0
    env_day = getattr(env, "day", None)

    cards = _cards_from_action_vec(env, np.asarray(action_vec, dtype=float), top_k=6)

    sim["step"] += 1
    sim["history"].append(
        {
            "step": sim["step"],
            "pv": pv,
            "reward": float(reward),
            "risk_penalty": risk_penalty,
            "env_day": env_day,
            "cards": cards,
        }
    )
    sim["done"] = bool(terminated or truncated)

# (İyileştirme #5) 10 gün ilerlet
def step_live_sim_n(n: int) -> int:
    sim = st.session_state.live_sim
    progressed = 0
    for _ in range(int(n)):
        if sim.get("done"):
            break
        step_live_sim()
        progressed += 1
    return progressed

# ----------------------------
# CLI pipeline (optional)
# ----------------------------
def run_main_cli(mode: str, profile: str = "all", timesteps: Optional[int] = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "src.main", "--mode", mode]
    if mode == "train":
        cmd += ["--profile", profile]
        if timesteps is not None:
            cmd += ["--timesteps", str(int(timesteps))]
    return subprocess.run(cmd, capture_output=True, text=True)

def main() -> None:
    st.set_page_config(page_title="RiskProfile RL – Portföy Koçu", layout="wide")
    inject_css()

    st.markdown(
        """
        <h1 style="font-weight:800; margin-bottom:0.15rem;">RiskProfile RL – Portföy Koçu</h1>
        <div style="opacity:0.82; margin-bottom:0.7rem;">
        Risk profiline uyarlanabilir RL ajanı • Backtest inceleme • Öneri kartları • Mini simülasyon
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.markdown("## Kontrol Paneli")
    profile = st.sidebar.radio(
        "Risk profili",
        ["conservative", "balanced", "aggressive"],
        index=1,
        format_func=_profile_label,
    )
    virtual_capital = st.sidebar.number_input(
        "Sanal sermaye (TL)",
        min_value=10_000,
        max_value=1_000_000,
        value=100_000,
        step=10_000,
    )

    # (İyileştirme #6) Developer mode toggle → Pipeline sekmesi gizli
    dev_mode = st.sidebar.toggle("Geliştirici modu (Pipeline)", value=False)
    st.sidebar.caption("B2C demo için kapalı tut. Eğitim/backtest için aç.")

    st.sidebar.markdown("---")
    st.sidebar.info("Bu uygulama eğitim/simülasyon amaçlıdır; yatırım tavsiyesi değildir.")

    ensure_virtual_portfolio(profile, virtual_capital)
    ensure_live_sim_state()

    # Data + results availability
    metrics_df = load_backtest_metrics()
    if metrics_df is None:
        st.warning(
            "`results/backtest_metrics.csv` bulunamadı. Önce `python -m src.main --mode data/train/backtest` çalıştırmalısın."
        )

    # (İyileştirme #1) Onboarding banner
    onboarding_banner(profile)
    st.markdown("")

    # Tabs (Pipeline sadece dev_mode ile)
    tab_names = ["📊 Genel Bakış", "📉 Karşılaştırma", "🤖 Öneri Kartları", "🎮 Mini Sim"]
    if dev_mode:
        tab_names.append("🛠 Pipeline")

    tabs = st.tabs(tab_names)
    tab_overview = tabs[0]
    tab_compare = tabs[1]
    tab_agent = tabs[2]
    tab_live = tabs[3]
    tab_pipeline = tabs[4] if dev_mode else None

    # ----------------------------
    # TAB: Overview
    # ----------------------------
    with tab_overview:
        st.subheader(f"Özet – {_profile_label(profile)}")

        # (İyileştirme #3) Sanal Portföy Özet Paneli
        st.markdown("### Sanal Portföy Özeti")
        summ = virtual_portfolio_summary()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Toplam İşlem", summ["total"])
        with c2:
            st.metric("BUY", summ["buy"])
        with c3:
            st.metric("SELL", summ["sell"])
        with c4:
            st.metric("HOLD", summ["hold"])

        if summ["last3"]:
            with st.expander("Son 3 işlem (detay)"):
                st.dataframe(pd.DataFrame(summ["last3"]), use_container_width=True)

        st.markdown("---")

        if metrics_df is not None:
            pcol = _find_profile_col(metrics_df)
            if pcol is None:
                st.error(f"Profil kolonu bulunamadı. Kolonlar: {list(metrics_df.columns)}")
            else:
                row = metrics_df[metrics_df[pcol].astype(str).str.lower() == profile].head(1)
                if row.empty:
                    st.warning(f"CSV içinde `{profile}` satırı yok. Var olanlar: {metrics_df[pcol].unique().tolist()}")
                else:
                    r = row.iloc[0]
                    tr = _find_metric(r, ["total_return_pct", "total_return", "return_pct"])
                    sh = _find_metric(r, ["sharpe", "sharpe_ratio"])
                    vol = _find_metric(r, ["volatility_pct", "volatility"])
                    mdd = _find_metric(r, ["max_drawdown_pct", "max_drawdown"])

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        metric_card("Toplam Getiri (%)", f"{tr:.2f}" if tr is not None else "N/A", profile, METRIC_HELP["total_return_pct"])
                    with c2:
                        metric_card("Sharpe", f"{sh:.2f}" if sh is not None else "N/A", profile, METRIC_HELP["sharpe"])
                    with c3:
                        metric_card("Volatilite (%)", f"{vol:.2f}" if vol is not None else "N/A", profile, METRIC_HELP["volatility_pct"])
                    with c4:
                        metric_card("Max Drawdown (%)", f"{mdd:.2f}" if mdd is not None else "N/A", profile, METRIC_HELP["max_drawdown_pct"])

        st.markdown("---")
        st.subheader("Portföy Değeri Zaman Serisi")
        series_df = load_series(profile)
        if series_df is None:
            st.caption(f"`results/series_{profile}.csv` yok. Backtest sonrası üretilir.")
        else:
            xcol = None
            for c in ["day", "step", "date", "datetime", "time"]:
                if c in series_df.columns:
                    xcol = c
                    break
            chart = series_df.sort_values(xcol).set_index(xcol) if xcol else series_df.copy()

            if "portfolio_value" in chart.columns:
                st.line_chart(chart[["portfolio_value"]], use_container_width=True)
            else:
                numeric_cols = chart.select_dtypes("number").columns.tolist()
                if numeric_cols:
                    st.line_chart(chart[numeric_cols], use_container_width=True)

        img = RESULTS_DIR / "backtest_comparison.png"
        if img.exists():
            st.markdown("---")
            st.subheader("Ajan vs Benchmark")
            st.image(str(img), use_container_width=True)

    # ----------------------------
    # TAB: Compare
    # ----------------------------
    with tab_compare:
        st.subheader("Profiller arası metrik karşılaştırması")
        if metrics_df is None:
            st.caption("Önce backtest üret.")
        else:
            st.dataframe(metrics_df, use_container_width=True)
            pcol = _find_profile_col(metrics_df)
            if pcol:
                metric_opt = st.selectbox(
                    "Metrik",
                    ["total_return_pct", "sharpe", "volatility_pct", "max_drawdown_pct"],
                    format_func=lambda x: x.replace("_pct", "").replace("_", " ").title(),
                )
                if metric_opt in metrics_df.columns:
                    plot_df = metrics_df[[pcol, metric_opt]].set_index(pcol)
                    st.bar_chart(plot_df, use_container_width=True)

        st.markdown("---")
        extra = RESULTS_DIR / "analysis_backtest_metrics.png"
        if extra.exists():
            st.image(str(extra), use_container_width=True)

    # ----------------------------
    # TAB: Agent cards
    # ----------------------------
    with tab_agent:
        st.subheader("Ajan Önerileri (kartlar)")
        st.caption("Model varsa PPO’dan üretir; yoksa demo kartlara düşer.")

        # (İyileştirme #3) Hızlı portföy özet satırı (kullanıcıyı motive eder)
        summ = virtual_portfolio_summary()
        st.info(f"Sanal portföyünde şu an **{summ['total']}** işlem var (BUY:{summ['buy']} / SELL:{summ['sell']} / HOLD:{summ['hold']}).")

        col_l, col_r = st.columns([2, 1])
        with col_r:
            if st.button("🔄 Önerileri yenile"):
                st.session_state.pop("cached_recs", None)

        if "cached_recs" not in st.session_state:
            try:
                st.session_state.cached_recs = [r.__dict__ for r in get_rl_recommendations(profile)]
                st.success("Öneriler PPO modelinden üretildi.")
            except Exception as e:
                st.session_state.cached_recs = [r.__dict__ for r in get_dummy_recommendations(profile)]
                st.warning(f"RL önerisi üretilemedi: {e}. Dummy kartlar gösteriliyor.")

        recs = [Recommendation(**d) for d in st.session_state.cached_recs]

        for i, rec in enumerate(recs):
            action = rec.action.upper()
            if action == "BUY":
                bg = "rgba(34,197,94,0.08)"; border = "rgba(34,197,94,0.45)"; pill = "rgba(34,197,94,0.16)"
            elif action == "SELL":
                bg = "rgba(239,68,68,0.08)"; border = "rgba(239,68,68,0.45)"; pill = "rgba(239,68,68,0.16)"
            else:
                bg = "rgba(148,163,184,0.10)"; border = "rgba(148,163,184,0.45)"; pill = "rgba(148,163,184,0.25)"

            # (İyileştirme #4) Kısa özet + Detay expander
            short_reason = _shorten(rec.reason, 90)
            short_risk = _shorten(rec.risk_note, 90)

            st.markdown(
                f"""
                <div class="rp-rec" style="background:{bg}; border:1px solid {border};">
                    <div class="rp-row">
                        <h4>{rec.ticker}</h4>
                        <span class="rp-pill" style="background:{pill};">{action}{'' if rec.amount is None else f' x {rec.amount}'}</span>
                    </div>
                    <div class="rp-muted" style="margin-top:6px;"><b>Neden (özet):</b> {short_reason}</div>
                    <div class="rp-muted"><b>Risk (özet):</b> {short_risk}</div>
                    <div class="rp-muted">Güç: {rec.strength:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Detay gör"):
                st.write(f"**Neden?** {rec.reason}")
                st.write(f"**Risk notu:** {rec.risk_note}")
                st.caption("Not: Bu açıklamalar kısa özet amaçlıdır (Explainability Lite).")

            c1, c2 = st.columns([1, 4])
            with c1:
                if st.button("✅ Sanal portföye ekle", key=f"accept_{profile}_{i}"):
                    accept_recommendation(rec, source="cards")
                    st.success("Eklendi.")
            with c2:
                st.caption("Eklenen işlemler 'Genel Bakış > Sanal Portföy Özeti'nde görünür.")

        st.markdown("---")
        st.subheader("📜 Sanal İşlem Geçmişi")
        actions = st.session_state.virtual_portfolio.get("actions", [])
        if not actions:
            st.caption("Henüz kabul edilmiş öneri yok.")
        else:
            st.dataframe(pd.DataFrame(actions), use_container_width=True)

        if st.button("🧹 Sanal portföyü sıfırla"):
            init_virtual_portfolio(profile, virtual_capital)
            st.success("Sıfırlandı.")

    # ----------------------------
    # TAB: Live sim
    # ----------------------------
    with tab_live:
        st.subheader("Mini Simülasyon (adım adım)")
        st.caption("Her adımda: portföy değeri + günlük skor + risk cezası + o günün öneri kartları.")

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            if st.button("🔁 Simülasyonu Başlat / Sıfırla", key="sim_init"):
                try:
                    init_live_simulation(profile)
                    st.success("Mini sim başlatıldı.")
                except Exception as e:
                    st.error(str(e))
        with col_b:
            if st.button("➡️ 1 Gün İlerlet", key="sim_step"):
                try:
                    step_live_sim()
                except Exception as e:
                    st.error(str(e))
        with col_c:
            # (İyileştirme #5) 10 gün ilerlet
            if st.button("⏩ 10 Gün İlerlet", key="sim_step10"):
                try:
                    progressed = step_live_sim_n(10)
                    st.success(f"{progressed} gün ilerletildi.")
                except Exception as e:
                    st.error(str(e))

        sim = st.session_state.live_sim
        if sim["env"] is None:
            st.caption("Önce simülasyonu başlat.")
        else:
            hist = sim["history"]
            if not hist:
                st.caption("Henüz adım yok.")
            else:
                steps = [h["step"] for h in hist]
                sel = st.selectbox("Gün seç (o günün kartları)", steps, index=len(steps) - 1)
                rec = next(h for h in hist if h["step"] == sel)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(LABELS["pv"], f"{rec.get('pv', 0):,.0f}")
                with m2:
                    st.metric(LABELS["reward"], f"{rec.get('reward', 0):.6f}")
                with m3:
                    st.metric(LABELS["risk_penalty"], f"{rec.get('risk_penalty', 0):.6f}")

                st.caption(f"{LABELS['env_day']}: {rec.get('env_day', '—')} | Sim bitti mi?: {sim.get('done', False)}")

                st.markdown("#### O Günün Öneri Kartları")
                cards = rec.get("cards", [])
                for j, c in enumerate(cards):
                    r = Recommendation(
                        ticker=str(c.get("ticker", "")),
                        action=str(c.get("action", "HOLD")),
                        strength=float(c.get("strength", 0.0)),
                        reason=str(c.get("reason", "—")),
                        risk_note=str(c.get("risk_note", "—")),
                        amount=c.get("amount"),
                    )

                    cols = st.columns([3, 1.2, 1.2])
                    with cols[0]:
                        st.write(f"**{r.ticker}** — {r.action} (güç={r.strength:.3f})")
                        st.caption(f"Neden (özet): {_shorten(r.reason, 110)}")
                        st.caption(f"Risk (özet): {_shorten(r.risk_note, 110)}")
                        with st.expander("Detay"):
                            st.write(f"**Neden?** {r.reason}")
                            st.write(f"**Risk:** {r.risk_note}")

                    with cols[1]:
                        if st.button("✅ Sanal portföye ekle", key=f"sim_accept_{sel}_{j}"):
                            accept_recommendation(r, source="mini_sim")
                            st.success("Eklendi.")
                    with cols[2]:
                        st.button("❌ Geç", key=f"sim_skip_{sel}_{j}")

                st.markdown("---")
                df_hist = pd.DataFrame(hist)
                st.line_chart(df_hist.set_index("step")[["pv"]], use_container_width=True)

                # Teknik olmayan kullanıcı için tabloyu sade tut
                st.dataframe(df_hist[["step", "pv", "reward", "risk_penalty", "env_day"]], use_container_width=True)

    # ----------------------------
    # TAB: Pipeline (Developer Mode)
    # ----------------------------
    if dev_mode and tab_pipeline is not None:
        with tab_pipeline:
            st.subheader("Pipeline (Geliştirici Modu)")
            st.caption("Bu bölüm demo kullanıcı için gizlenir. Eğitim/backtest opsiyonları buradadır.")

            timesteps = st.number_input(
                "Train timesteps",
                min_value=1_000,
                max_value=2_000_000,
                step=1_000,
                value=int(getattr(config, "PPO_TOTAL_TIMESTEPS", 10_000)),
            )
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("1) Veri Hazırla"):
                    with st.spinner("Veri hazırlanıyor..."):
                        try:
                            os.makedirs(DATA_DIR, exist_ok=True)
                            DataProcessor().run()
                            load_csv.clear()
                            st.success("Veri hazır.")
                        except Exception as e:
                            st.error(str(e))

            with col2:
                if st.button("2) Eğit"):
                    with st.spinner("Eğitiliyor..."):
                        try:
                            df = load_csv(Path(config.DATA_SAVE_PATH))
                            if df is None:
                                raise RuntimeError("Önce veri üret.")
                            old = getattr(config, "PPO_TOTAL_TIMESTEPS", None)
                            config.PPO_TOTAL_TIMESTEPS = int(timesteps)
                            AgentTrainer(df).train_agent(profile)
                            if old is not None:
                                config.PPO_TOTAL_TIMESTEPS = old
                            st.success("Eğitim tamam.")
                        except Exception as e:
                            st.error(str(e))

            with col3:
                if st.button("3) Backtest"):
                    with st.spinner("Backtest..."):
                        try:
                            Backtester().run_backtest()
                            load_csv.clear()
                            st.success("Backtest tamam. Sekmelerde sonuçlar güncellendi.")
                        except Exception as e:
                            st.error(str(e))

            st.markdown("---")
            st.caption("Terminalden çalıştırmak istersen:")
            st.code(
                "python -m src.main --mode data\n"
                "python -m src.main --mode train --profile all --timesteps 50000\n"
                "python -m src.main --mode backtest",
                language="bash",
            )


if __name__ == "__main__":
    main()
