import pandas as pd
import numpy as np

INPUT_PATH = "stock_data/ml_dataset.parquet"
SAVE_PATH = "stock_data/signals.parquet"

df = pd.read_parquet(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# =========================
# cross-sectional rank（統一）
# =========================
g = df.groupby("Date")

df["ret_rank"] = g["forward_return"].rank(pct=True)
df["vol_rank"] = g["volume_ratio_5d"].rank(pct=True)
df["trend_rank"] = g["close_ma5_ratio"].rank(pct=True)
df["bb_rank"] = g["bb_position"].rank(pct=True)
df["risk_rank"] = 1 - g["atr_ratio"].rank(pct=True)

# =========================
# ① TREND LAYER（フィルタ）
# =========================
df["trend_filter"] = (
    (df["close_ma5_ratio"] > 1.01) &
    (df["close_ma25_ratio"] > 1.00) &
    (df["ma25_slope"] > 0)
).astype(int)

# =========================
# ② MOMENTUM LAYER（核）
# =========================
df["momentum_score"] = (
    0.6 * df["ret_rank"] +
    0.4 * df["vol_rank"]
)

# =========================
# ③ TIMING LAYER（入口精度）
# =========================
df["timing_score"] = (
    0.5 * df["bb_rank"] +
    0.5 * df["high_break_20d"]
)

# volatility / risk
df["risk_score"] = (
    0.5 * df["risk_rank"] +
    0.5 * (1 - df["gap_up_ratio"].clip(0, 0.1))
)

# =========================
# ④ FINAL ALPHA（構造型）
# =========================
df["alpha_score"] = (
    0.5 * df["momentum_score"] +
    0.3 * df["timing_score"] +
    0.2 * df["risk_score"]
)

# =========================
# ⑤ クロスセクション正規化（重要：1回だけ）
# =========================
df["alpha_score"] = g["alpha_score"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-9)
)

# =========================
# ⑥ 3層シグナル（本体）
# =========================

# strong（条件型：方向 + 上位）
df["signal_strong"] = (
    (df["trend_filter"] == 1) &
    (df["alpha_score"] > 1.0) &
    (df["ret_rank"] > 0.75)
).astype(int)

# weak（準エッジ：トレンドあり + 中位以上）
df["signal_weak"] = (
    (df["trend_filter"] == 1) &
    (df["alpha_score"] > 0.2) &
    (df["signal_strong"] == 0)
).astype(int)

# noise filter（重要）
df["signal_entry"] = (
    df["signal_strong"] * 2 +
    df["signal_weak"]
)

df["signal_trade"] = (df["signal_entry"] >= 1).astype(int)

# =========================
# ⑦ sanity check
# =========================
print("\n===== SIGNAL DISTRIBUTION =====")
print(df["signal_entry"].value_counts())

print("\n===== STRONG RATE =====")
print(df["signal_strong"].mean())

print("\n===== WEAK RATE =====")
print(df["signal_weak"].mean())

print("\n===== ALPHA STATS =====")
print(df["alpha_score"].describe())

# =========================
# save
# =========================
df.to_parquet(SAVE_PATH, index=False)

print("\nSaved:", SAVE_PATH)