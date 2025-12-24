import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

from src.data_processor import DataProcessor
from src.train import AgentTrainer
from src.backtest import Backtester
import config


RESULTS_DIR = "./results"
DATA_DIR = "./data"


def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def list_series_files():
    pattern = os.path.join(RESULTS_DIR, "series_*.csv")
    files = glob.glob(pattern)
    items = []
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0].replace("series_", "")
        items.append((name, p))
    return items


def ensure_data_ready():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(config.DATA_SAVE_PATH):
        processor = DataProcessor()
        df = processor.run()
        return df
    df = pd.read_csv(config.DATA_SAVE_PATH)
    return df


def run_interactive_train_and_backtest(selected_profile, timesteps):
    df = ensure_data_ready()
    original_timesteps = getattr(config, "PPO_TOTAL_TIMESTEPS", None)
    original_model_dir = config.TRAINED_MODEL_DIR
    config.PPO_TOTAL_TIMESTEPS = int(timesteps)
    config.TRAINED_MODEL_DIR = "./models_streamlit"
    os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)
    trainer = AgentTrainer(df)
    trainer.train_agent(selected_profile)
    backtester = Backtester()
    backtester.run_backtest()
    if original_timesteps is not None:
        config.PPO_TOTAL_TIMESTEPS = original_timesteps
    config.TRAINED_MODEL_DIR = original_model_dir


def list_profiles_from_backtest(df_backtest):
    if df_backtest is None:
        return []
    if "profile" not in df_backtest.columns:
        return []
    return df_backtest["profile"].astype(str).tolist()


def main():
    st.set_page_config(page_title="Risk-Duyarlı RL Portföy Dashboard", layout="wide")

    st.sidebar.title("Navigasyon")
    section = st.sidebar.selectbox(
        "Bölüm seç",
        [
            "Genel Bakış",
            "Backtest & Profiller",
            "Zaman Serileri",
            "Ablation Çalışması",
            "Model Analizi",
            "Canlı Eğitim & Backtest",
        ],
    )

    df_backtest = load_csv(os.path.join(RESULTS_DIR, "backtest_metrics.csv"))
    df_ablation = load_csv(os.path.join(RESULTS_DIR, "ablation_metrics.csv"))

    if section == "Genel Bakış":
        st.title("Risk-Duyarlı RL Portföy Dashboard")
        st.write("Bu panel, eğitilmiş PPO ajanlarının performansını, risk profillerini ve ablation deneylerini incelemek için tasarlanmıştır.")
        cols = st.columns(3)
        with cols[0]:
            st.subheader("Veri Dosyaları")
            data_files = glob.glob(os.path.join(DATA_DIR, "processed*.csv"))
            if data_files:
                for p in data_files:
                    st.write(os.path.basename(p))
            else:
                st.write("Hazırlanmış veri dosyası bulunamadı.")
        with cols[1]:
            st.subheader("Backtest Sonuçları")
            if df_backtest is not None:
                st.dataframe(df_backtest)
            else:
                st.write("backtest_metrics.csv bulunamadı.")
        with cols[2]:
            st.subheader("Ablation Sonuçları")
            if df_ablation is not None:
                st.dataframe(df_ablation)
            else:
                st.write("ablation_metrics.csv bulunamadı.")
        st.markdown("---")
        st.subheader("Genel Backtest Grafiği")
        img_path = os.path.join(RESULTS_DIR, "backtest_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)
        else:
            st.write("backtest_comparison.png bulunamadı.")

    elif section == "Backtest & Profiller":
        st.title("Backtest & Risk Profilleri")
        if df_backtest is None:
            st.write("backtest_metrics.csv bulunamadı.")
        else:
            st.subheader("Metrik Tablosu")
            st.dataframe(df_backtest)
            st.subheader("Toplam Getiri / Sharpe / Max Drawdown")
            metric_option = st.selectbox(
                "Metrik seç",
                ["total_return_pct", "sharpe", "max_drawdown_pct"],
                format_func=lambda x: {
                    "total_return_pct": "Toplam Getiri (%)",
                    "sharpe": "Sharpe",
                    "max_drawdown_pct": "Max Drawdown (%)",
                }.get(x, x),
            )
            chart_data = df_backtest[["profile", metric_option]].set_index("profile")
            st.bar_chart(chart_data)
        st.markdown("---")
        st.subheader("Kümülatif Getiri Grafiği")
        img_path = os.path.join(RESULTS_DIR, "backtest_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)
        else:
            st.write("backtest_comparison.png bulunamadı.")
        st.markdown("---")
        st.subheader("Ek Metrik Grafiği")
        img2 = os.path.join(RESULTS_DIR, "analysis_backtest_metrics.png")
        if os.path.exists(img2):
            st.image(img2, use_column_width=True)
        else:
            st.write("analysis_backtest_metrics.png bulunamadı.")

    elif section == "Zaman Serileri":
        st.title("Zaman Serileri – Profil Bazında")
        series_files = list_series_files()
        if not series_files:
            st.write("series_*.csv bulunamadı.")
        else:
            names = [n for n, _ in series_files]
            selected_name = st.selectbox("Profil seç", names)
            path = dict(series_files)[selected_name]
            df_series = load_csv(path)
            if df_series is None:
                st.write("Seçili seri dosyası boş veya okunamadı.")
            else:
                st.subheader(f"Portföy Değeri – {selected_name}")
                if "day" in df_series.columns and "portfolio_value" in df_series.columns:
                    chart_df = df_series[["day", "portfolio_value"]].set_index("day")
                    st.line_chart(chart_df)
                st.markdown("---")
                cols = st.columns(2)
                with cols[0]:
                    if "raw_reward" in df_series.columns:
                        st.subheader("Ham Ödül")
                        rr = df_series[["day", "raw_reward"]].set_index("day")
                        st.line_chart(rr)
                with cols[1]:
                    if "risk_adjusted_reward" in df_series.columns:
                        st.subheader("Risk-Adjust Ödül")
                        ar = df_series[["day", "risk_adjusted_reward"]].set_index("day")
                        st.line_chart(ar)
                st.markdown("---")
                if "volatility_metric" in df_series.columns:
                    st.subheader("Volatilite Göstergesi")
                    vol = df_series[["day", "volatility_metric"]].set_index("day")
                    st.line_chart(vol)
        st.markdown("---")
        st.subheader("Hazır Seri Grafikleri (PNG)")
        for name in ["conservative", "balanced", "aggressive"]:
            img_path = os.path.join(RESULTS_DIR, f"analysis_series_{name}.png")
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{name}", use_column_width=True)

    elif section == "Ablation Çalışması":
        st.title("Ablation Çalışması – İndikatör Kombinasyonları")
        if df_ablation is None:
            st.write("ablation_metrics.csv bulunamadı.")
        else:
            st.subheader("Metrik Tablosu")
            st.dataframe(df_ablation)
            st.markdown("---")
            st.subheader("Sharpe Karşılaştırması")
            sharpe_df = df_ablation[["experiment", "sharpe"]].set_index("experiment")
            st.bar_chart(sharpe_df)
            st.markdown("---")
            st.subheader("Risk-Getiri Diyagramı")
            if {"volatility_pct", "total_return_pct"}.issubset(df_ablation.columns):
                st.scatter_chart(
                    df_ablation.rename(
                        columns={
                            "volatility_pct": "Volatility",
                            "total_return_pct": "TotalReturn",
                        }
                    ),
                    x="Volatility",
                    y="TotalReturn",
                )
            st.markdown("---")
            st.subheader("PNG Grafikler")
            img1 = os.path.join(RESULTS_DIR, "analysis_ablation_sharpe.png")
            img2 = os.path.join(RESULTS_DIR, "analysis_ablation_risk_return.png")
            cols = st.columns(2)
            with cols[0]:
                if os.path.exists(img1):
                    st.image(img1, use_column_width=True)
            with cols[1]:
                if os.path.exists(img2):
                    st.image(img2, use_column_width=True)

    elif section == "Model Analizi":
        st.title("Model Analizi – Policy Davranışı")
        st.write("Aşağıdaki grafikler, eğitilmiş PPO ajanlarının politika ağırlıklarını, aksiyon dağılımını ve fiyat duyarlılığını göstermektedir.")
        profiles = ["conservative", "balanced", "aggressive"]
        selected_profile = st.selectbox("Profil seç", profiles)
        col1, col2 = st.columns(2)
        with col1:
            img_w = os.path.join(RESULTS_DIR, f"model_weights_{selected_profile}.png")
            if os.path.exists(img_w):
                st.subheader("Policy Weight Distribution")
                st.image(img_w, use_column_width=True)
            else:
                st.write(f"model_weights_{selected_profile}.png bulunamadı.")
        with col2:
            img_p = os.path.join(RESULTS_DIR, f"model_policy_dist_{selected_profile}.png")
            if os.path.exists(img_p):
                st.subheader("Action Mean per Asset")
                st.image(img_p, use_column_width=True)
            else:
                st.write(f"model_policy_dist_{selected_profile}.png bulunamadı.")
        st.markdown("---")
        img_s = os.path.join(RESULTS_DIR, f"model_action_sensitivity_{selected_profile}.png")
        if os.path.exists(img_s):
            st.subheader("Action Sensitivity to Price Changes")
            st.image(img_s, use_column_width=True)
        else:
            st.write(f"model_action_sensitivity_{selected_profile}.png bulunamadı.")

    elif section == "Canlı Eğitim & Backtest":
        st.title("Canlı Eğitim & Backtest")
        st.write("Buradan seçtiğiniz risk profili ve timestep değeri ile modeli yeniden eğitip aynı seans içinde backtest edebilirsiniz.")
        profiles = ["conservative", "balanced", "aggressive"]
        sel_profile = st.selectbox("Profil seç", profiles, index=1)
        default_ts = int(getattr(config, "PPO_TOTAL_TIMESTEPS", 10000))
        timesteps = st.number_input("Toplam timesteps", min_value=1000, max_value=2000000, step=1000, value=default_ts)
        if st.button("Eğit ve Backtest Yap"):
            with st.spinner("Model eğitiliyor ve backtest yapılıyor... Bu işlem birkaç dakika sürebilir."):
                run_interactive_train_and_backtest(sel_profile, timesteps)
            st.success("Eğitim ve backtest tamamlandı. Sonuçlar Backtest & Profiller sekmesinde güncellendi.")
            new_df = load_csv(os.path.join(RESULTS_DIR, "backtest_metrics.csv"))
            if new_df is not None:
                st.subheader("Güncel Backtest Metrikleri")
                st.dataframe(new_df)


if __name__ == "__main__":
    main()
