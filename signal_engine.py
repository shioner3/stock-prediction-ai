import pandas as pd
import numpy as np

# =========================
# 設定
# =========================
INPUT_PATH = "stock_data/ml_dataset.parquet"
SAVE_PATH = "stock_data/signals.parquet"

# =========================
# 読み込み
# =========================
df = pd.read_parquet(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# =========================
# BB（安全版）
# =========================
bb_ma = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
bb_std = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).std())

bb_upper = bb_ma + 2 * bb_std
bb_lower = bb_ma - 2 * bb_std

df["bb_width"] = (bb_upper - bb_lower) / bb_ma

# =========================
# シグナル（0/1）
# =========================
df["sig_trend"] = (
    (df["close_ma5_ratio"] > 1.01) &
    (df["close_ma25_ratio"] > 1.00) &
    (df["ma25_slope"] > 0)
).astype(int)

df["sig_breakout"] = (
    (df["high_break_20d"] == 1)
).astype(int)

df["sig_momentum"] = (
    (df["return_5d"] > 0)
).astype(int)

df["sig_volume"] = (
    (df["volume_ratio_5d"] > 1.05) &
    (df["volume_rank_daily"] > 0.4)
).astype(int)

df["sig_low_volatility_entry"] = (
    (df["atr_ratio"] < 0.12)
).astype(int)

bb_mean_50 = df.groupby("Ticker")["bb_width"].transform(lambda x: x.rolling(50).mean())

df["sig_bb_setup"] = (
    (df["bb_position"] > 0.65) &
    (df["bb_width"] < bb_mean_50)
).astype(int)

df["sig_gap_support"] = (
    (df["gap_up_ratio"] > 0) &
    (df["gap_up_ratio"] < 0.05)
).astype(int)

df["sig_intraday_strength"] = (
    (df["return_rank_daily"] > 0.6) &
    (df["volume_rank_daily"] > 0.5)
).astype(int)

# =========================
# 🔥 スコア（分散を作る核心）
# =========================
df["signal_score"] = (
    df["sig_trend"] * 2.5 +
    df["sig_volume"] * 2.0 +
    df["sig_intraday_strength"] * 2.0 +
    df["sig_breakout"] * 1.5 +
    df["sig_momentum"] * 1.5 +
    df["sig_low_volatility_entry"] * 1.0 +
    df["sig_bb_setup"] * 1.0 +
    df["sig_gap_support"] * 0.8
)

# =========================
# 🔥 分散補正（重要）
# → 横並びにならないように日次標準化
# =========================
df["signal_score"] = df.groupby("Date")["signal_score"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-9)
)

# =========================
# 🔥 strong（緩める）
# =========================
df["signal_strong"] = (
    (df["signal_score"] > 1.0) &   # ← 閾値型に変更
    (df["sig_trend"] == 1) &
    (df["sig_volume"] == 1)
).astype(int)

# =========================
# 🔥 weak（意味あるものに）
# =========================
df["signal_weak"] = (
    (df["signal_score"] > 0.3) &      # 初動だけ拾う
    (df["signal_score"] <= 1.0) &
    (df["sig_volume"] == 1)           # 出来高必須
).astype(int)

# =========================
# 最終エントリー
# =========================
df["signal_entry"] = (
    df["signal_strong"] * 2 +
    df["signal_weak"] * 1
)

df["signal_trade"] = (df["signal_entry"] >= 1).astype(int)

# =========================
# 分布確認
# =========================
print("\n===== SIGNAL DISTRIBUTION =====")
print(df["signal_entry"].value_counts())

print("\n===== STRONG =====")
print(df["signal_strong"].value_counts())

print("\n===== WEAK =====")
print(df["signal_weak"].value_counts())

print("\n===== SCORE STATS =====")
print(df["signal_score"].describe())

# =========================
# 保存
# =========================
df.to_parquet(SAVE_PATH, index=False)

print("\nSaved to:", SAVE_PATH)