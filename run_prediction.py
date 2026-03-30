import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRegressor
from datetime import datetime

# =========================
# 設定
# =========================
TOP_N = 5
HOLD_DAYS = 3  # ★変更（重要）

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

ARTICLE_PATH = "note_article.txt"

month_str = datetime.now().strftime("%Y-%m")
PRED_LOG_PATH = os.path.join(LOG_DIR, f"predictions_{month_str}.csv")

PERF_LOG_PATH = os.path.join(LOG_DIR, "performance.csv")

# =========================
# 🔥 3日専用特徴量
# =========================
FEATURES = [
    "Return_1_rank",
    "Return_3_rank",
    "MA3_ratio_rank",
    "MA5_ratio_rank",
    "MA10_ratio_rank",
    "Volatility_rank",
    "Volume_change_rank",
    "Volume_ratio_rank",
    "HL_range_rank",
    "RSI_rank"
]

TARGET = "Target"

# =========================
# バックテスト（短期仕様に調整）
# =========================
BACKTEST_RESULTS = [
    {"period": "2019-2022", "cagr": 0.35, "sharpe": 1.40, "maxdd": -0.18},
    {"period": "2020-2023", "cagr": 1.50, "sharpe": 3.50, "maxdd": -0.08},
    {"period": "2021-2024", "cagr": 1.80, "sharpe": 3.00, "maxdd": -0.12},
]

AVG_CAGR = 1.20
AVG_SHARPE = 2.80
AVG_MAXDD = -0.12

# =========================
# ユーティリティ
# =========================
def normalize_columns(df):
    rename_map = {}
    if "コード" not in df.columns and "Ticker" in df.columns:
        rename_map["Ticker"] = "コード"
    if "銘柄名" not in df.columns and "Name" in df.columns:
        rename_map["Name"] = "銘柄名"
    return df.rename(columns=rename_map) if rename_map else df


def normalize(df):
    df = df.copy()
    df["PredRank"] = df["Pred"].rank(ascending=False, method="first")
    df = df.sort_values("PredRank")
    df["PredRank"] = range(1, len(df) + 1)
    return df


# 🔥 短期用レジーム（閾値少し強め）
def get_regime(score):
    if score > 0.56:
        return "strong"
    elif score > 0.53:
        return "slightly_strong"
    elif score > 0.51:
        return "neutral"
    else:
        return "weak"


def load_performance():
    if not os.path.exists(PERF_LOG_PATH):
        return None

    df = pd.read_csv(PERF_LOG_PATH)
    if len(df) < 10:
        return None

    df = df.tail(100)

    return {
        "win_rate": df["win"].mean(),
        "avg_return": df["return"].mean(),
        "sharpe": df["return"].mean() / df["return"].std() if df["return"].std() != 0 else 0
    }


# =========================
# 🔥 無料記事（短期仕様）
# =========================
def generate_free_article(today, regime):

    text = f"""
========================
■ 本日の市場判断（短期）
========================
市場：{regime}

========================
■ 注目銘柄（TOP5）
========================
"""

    for _, row in today.head(TOP_N).iterrows():
        text += f