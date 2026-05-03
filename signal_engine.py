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
# シグナル定義（0/1）
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
    (df["bb_width"] < df["bb_width"].rolling(50).mean())
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
# 🔥 シグナル強度（2段階化）
# =========================
# 0 = no signal
# 1 = weak signal（部分一致）
# 2 = strong signal（完全一致）

df["signal_score"] = (
    df["sig_trend"] * 2 +
    df["sig_breakout"] * 2 +
    df["sig_momentum"] * 1.5 +
    df["sig_volume"] * 1.5 +
    df["sig_low_volatility_entry"] * 1.0 +
    df["sig_bb_setup"] * 1.0 +
    df["sig_gap_support"] * 1.0 +
    df["sig_intraday_strength"] * 2
)

# =========================
# 強度2段階化
# =========================
# 強い条件：複数コア同時成立
df["signal_strong"] = (
    (df["sig_trend"] == 1) &
    (df["sig_volume"] == 1) &
    (df["sig_intraday_strength"] == 1) &
    (df["signal_score"] >= 6)
).astype(int)

# 弱いシグナル：どれか刺さる
df["signal_weak"] = (
    (df["signal_score"] >= 3) &
    (df["signal_strong"] == 0)
).astype(int)

# エントリー統合
df["signal_entry"] = (
    df["signal_strong"] * 2 +
    df["signal_weak"] * 1
)

# =========================
# フィルタ（実運用用）
# =========================
df["signal_trade"] = (df["signal_entry"] >= 1).astype(int)

# =========================
# 検証用
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