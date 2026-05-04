import pandas as pd
import numpy as np

INPUT_PATH = "stock_data/ml_dataset.parquet"
SAVE_PATH = "stock_data/signals.parquet"

df = pd.read_parquet(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# =========================
# ★ 市場レジーム（追加）
# =========================
# market_return は前工程で作成済み前提
# なければここで作る必要あり

df["market_trend"] = (df["market_return"] > 0).astype(int)

# 強さ（連続値）
df["market_strength"] = df["market_return"]

# =========================
# cross-sectional rank
# =========================
g = df.groupby("Date")

df["ret_rank"] = g["forward_return"].rank(pct=True)
df["vol_rank"] = g["volume_ratio_5d"].rank(pct=True)
df["trend_rank"] = g["close_ma5_ratio"].rank(pct=True)
df["bb_rank"] = g["bb_position"].rank(pct=True)
df["risk_rank"] = 1 - g["atr_ratio"].rank(pct=True)

# =========================
# ① TREND LAYER
# =========================
df["trend_filter"] = (
    (df["close_ma5_ratio"] > 1.01) &
    (df["close_ma25_ratio"] > 1.00) &
    (df["ma25_slope"] > 0)
).astype(int)

# ★ 市場フィルタ追加（超重要）
df["trend_filter"] = df["trend_filter"] * df["market_trend"]

# =========================
# ② MOMENTUM
# =========================
df["momentum_score"] = (
    0.6 * df["ret_rank"] +
    0.4 * df["vol_rank"]
)

# =========================
# ③ TIMING
# =========================
df["timing_score"] = (
    0.5 * df["bb_rank"] +
    0.5 * df["high_break_20d"]
)

# =========================
# ④ RISK
# =========================
df["risk_score"] = (
    0.5 * df["risk_rank"] +
    0.5 * (1 - df["gap_up_ratio"].clip(0, 0.1))
)

# =========================
# ⑤ FINAL SCORE
# =========================
df["signal_score"] = (
    0.5 * df["momentum_score"] +
    0.3 * df["timing_score"] +
    0.2 * df["risk_score"]
)

# =========================
# ★ 市場強度で重み付け（ここが効く）
# =========================
df["signal_score"] = df["signal_score"] * (1 + df["market_strength"])

# =========================
# 正規化
# =========================
df["signal_score"] = g["signal_score"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-9)
)

# =========================
# alpha
# =========================
df["alpha_score"] = df["signal_score"]

# =========================
# ⑥ シグナル
# =========================

df["signal_strong"] = (
    (df["trend_filter"] == 1) &
    (df["alpha_score"] > 1.0) &
    (df["ret_rank"] > 0.75)
).astype(int)

df["signal_weak"] = (
    (df["trend_filter"] == 1) &
    (df["alpha_score"] > 0.2) &
    (df["signal_strong"] == 0)
).astype(int)

df["signal_entry"] = (
    df["signal_strong"] * 2 +
    df["signal_weak"]
)

df["signal_trade"] = (df["signal_entry"] >= 1).astype(int)

# =========================
# sanity check
# =========================
print("\n===== SIGNAL DISTRIBUTION =====")
print(df["signal_entry"].value_counts())

print("\n===== STRONG RATE =====")
print(df["signal_strong"].mean())

print("\n===== WEAK RATE =====")
print(df["signal_weak"].mean())

print("\n===== SIGNAL_SCORE STATS =====")
print(df["signal_score"].describe())

# =========================
# save
# =========================
df.to_parquet(SAVE_PATH, index=False)

print("\nSaved:", SAVE_PATH)