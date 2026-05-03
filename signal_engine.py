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
# 安全なBB（groupby修正）
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
    (df["close_ma5_ratio"] > 1.02) &
    (df["close_ma25_ratio"] > 1.00) &
    (df["ma25_slope"] > 0)
).astype(int)

df["sig_breakout"] = (
    (df["high_break_20d"] == 1) &
    (df["return_5d"] > 0)
).astype(int)

df["sig_momentum"] = (
    (df["return_5d"] > 0.03) &
    (df["return_20d"] > 0)
).astype(int)

df["sig_volume"] = (
    (df["volume_ratio_5d"] > 1.1) &
    (df["volume_ratio_20d"] > 1.0) &
    (df["volume_zscore"] > 0)
).astype(int)

df["sig_low_volatility_entry"] = (
    (df["atr_ratio"] < 0.12) &
    (df["range_compression_5d"] < 1.05)
).astype(int)

df["sig_bb_setup"] = (
    (df["bb_position"] > 0.7) &
    (df["bb_width"] < df["bb_width"].rolling(50, min_periods=10).mean())
).astype(int)

df["sig_gap_support"] = (
    (df["gap_up_ratio"] > 0) &
    (df["gap_up_ratio"] < 0.05)
).astype(int)

df["sig_intraday_strength"] = (
    (df["return_rank_daily"] > 0.7) &
    (df["volume_rank_daily"] > 0.5)
).astype(int)

# =========================
# コア構造スコア（軽量化）
# =========================
df["core_score"] = (
    df["sig_trend"] * 2 +
    df["sig_volume"] * 2 +
    df["sig_intraday_strength"] * 2 +
    df["sig_breakout"] * 2 +
    df["sig_momentum"] * 1
)

# =========================
# 強シグナル（構造ベース）
# =========================
df["signal_strong"] = (
    (df["sig_trend"] == 1) &
    (df["sig_volume"] == 1) &
    (df["sig_intraday_strength"] == 1) &
    (df["core_score"] >= 6)
).astype(int)

# =========================
# 弱シグナル（補助）
# =========================
df["signal_weak"] = (
    (df["core_score"] >= 3) &
    (df["signal_strong"] == 0)
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

print("\n===== STRONG SIGNAL =====")
print(df["signal_strong"].value_counts())

print("\n===== WEAK SIGNAL =====")
print(df["signal_weak"].value_counts())

# =========================
# 保存
# =========================
df.to_parquet(SAVE_PATH, index=False)

print("\nSaved to:", SAVE_PATH)