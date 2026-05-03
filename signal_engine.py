import pandas as pd
import numpy as np

# =========================
# 設定
# =========================
INPUT_PATH = "stock_data/technical_features.parquet"
SAVE_PATH = "stock_data/signals.parquet"

# =========================
# 読み込み
# =========================
df = pd.read_parquet(INPUT_PATH)

df["Date"] = pd.to_datetime(df["Date"])

# =========================
# シグナル定義（コア）
# =========================
# 各シグナルは「1 or 0」で持つ

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
# 総合スコア（最重要）
# =========================
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
# エントリーシグナル
# =========================
df["signal_entry"] = (df["signal_score"] >= 4).astype(int)

# 上位フィルタ（強いシグナル）
df["signal_strong_entry"] = (df["signal_score"] >= 6).astype(int)

# =========================
# シグナル品質確認用
# =========================
print("\n===== SIGNAL SUMMARY =====")
print(df["signal_entry"].value_counts())

print("\n===== STRONG SIGNAL =====")
print(df["signal_strong_entry"].value_counts())

# =========================
# 保存
# =========================
df.to_parquet(SAVE_PATH, index=False)

print("\nSaved to:", SAVE_PATH)