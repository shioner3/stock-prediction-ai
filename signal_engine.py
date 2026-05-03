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
# feature compression
# =========================
df["trend"] = (
    0.5 * df["close_ma5_ratio"] +
    0.5 * df["close_ma25_ratio"]
)

df["momentum"] = (
    0.6 * df["ret_rank"] +
    0.4 * df["vol_rank"]
)

df["breakout"] = (
    0.5 * df["high_break_20d"] +
    0.5 * df["bb_rank"]
)

df["risk"] = (
    0.5 * df["risk_rank"] +
    0.5 * (1 - df["gap_up_ratio"].clip(0, 0.1))
)

# =========================
# normalize（重要）
# =========================
for c in ["trend", "momentum", "breakout", "risk"]:
    df[c] = g[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

# =========================
# signal score
# =========================
df["signal_score"] = (
    0.35 * df["trend"] +
    0.30 * df["momentum"] +
    0.20 * df["breakout"] +
    0.15 * df["risk"]
)

df["signal_score"] = g["signal_score"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-9)
)

# =========================
# strong / weak（安定版）
# =========================
df["signal_strong"] = (
    (df["signal_score"] > 1.0) &
    (df["ret_rank"] > 0.75)
).astype(int)

df["signal_weak"] = (
    (df["signal_score"] > 0.2) &
    (df["signal_strong"] == 0)
).astype(int)

# =========================
# entry
# =========================
df["signal_entry"] = (
    df["signal_strong"] * 2 +
    df["signal_weak"]
)

df.to_parquet(SAVE_PATH, index=False)

print("Saved:", SAVE_PATH)