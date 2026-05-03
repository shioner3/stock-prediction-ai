import pandas as pd
import numpy as np

INPUT_PATH = "stock_data/technical_features.parquet"
SAVE_PATH = "stock_data/signals.parquet"

df = pd.read_parquet(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# =========================
# 🔥 0. 防御（列ゆれ対策）
# =========================
if "volume_rank" not in df.columns:
    df["volume_rank"] = df.groupby("Date")["volume_ratio_5d"].rank(pct=True)

if "bb_rank" not in df.columns:
    df["bb_rank"] = df.groupby("Date")["bb_position"].rank(pct=True)

# =========================
# 🔥 ① クロスセクション正規化
# =========================
df["ret_rank"] = df.groupby("Date")["return_5d"].rank(pct=True)
df["volume_rank_daily"] = df.groupby("Date")["volume_ratio_5d"].rank(pct=True)
df["bb_rank"] = df.groupby("Date")["bb_position"].rank(pct=True)
df["trend_rank"] = df.groupby("Date")["close_ma5_ratio"].rank(pct=True)

df["inv_vol_rank"] = 1 - df.groupby("Date")["atr_ratio"].rank(pct=True)

# =========================
# 🔥 ② トレンドスコア
# =========================
df["trend_score"] = (
    0.4 * df["close_ma5_ratio"].clip(0.8, 1.2) +
    0.4 * df["close_ma25_ratio"].clip(0.8, 1.2) +
    0.2 * df["bb_position"]
)

df["trend_score"] = df.groupby("Date")["trend_score"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-9)
)

# =========================
# 🔥 ③ モメンタムスコア
# =========================
df["momentum_score"] = (
    0.6 * df["ret_rank"] +
    0.4 * df["volume_rank_daily"]   # ←ここ統一（重要）
)

# =========================
# 🔥 ④ ブレイクアウト
# =========================
df["break_score"] = (
    df["high_break_20d"] * 0.5 +
    df["bb_rank"] * 0.5
)

# =========================
# 🔥 ⑤ リスク
# =========================
df["risk_score"] = (
    df["inv_vol_rank"] * 0.5 +
    (1 - df["gap_up_ratio"].clip(0, 0.1)) * 0.5
)

# =========================
# 🔥 ⑥ アルファスコア
# =========================
df["signal_score"] = (
    0.35 * df["trend_score"] +
    0.30 * df["momentum_score"] +
    0.20 * df["break_score"] +
    0.15 * df["risk_score"]
)

# =========================
# 🔥 ⑦ 分位シグナル
# =========================
df["signal_strong"] = (
    df.groupby("Date")["signal_score"]
    .transform(lambda x: x > x.quantile(0.8))
).astype(int)

df["signal_weak"] = (
    (df["signal_score"] >
     df.groupby("Date")["signal_score"].transform("median")) &
    (df["signal_strong"] == 0)
).astype(int)

# =========================
# 🔥 ⑧ エントリー
# =========================
df["signal_entry"] = (
    df["signal_strong"] * 2 +
    df["signal_weak"] * 1
)

df["signal_trade"] = (df["signal_entry"] >= 1).astype(int)

# =========================
# 🔥 ⑨ チェック
# =========================
print("\n===== SIGNAL DISTRIBUTION =====")
print(df["signal_entry"].value_counts())

print("\n===== SCORE STATS =====")
print(df["signal_score"].describe())

print("\n===== STRONG RATE =====")
print(df["signal_strong"].mean())

print("\n===== WEAK RATE =====")
print(df["signal_weak"].mean())

# =========================
# save
# =========================
df.to_parquet(SAVE_PATH, index=False)

print("\nSaved:", SAVE_PATH)