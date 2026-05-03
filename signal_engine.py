import pandas as pd
import numpy as np

INPUT_PATH = "stock_data/technical_features.parquet"
SAVE_PATH = "stock_data/signals.parquet"

df = pd.read_parquet(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# =========================================================
# ① クロスセクション正規化（全ての基盤）
# =========================================================
g = df.groupby("Date")

df["ret_rank"] = g["return_5d"].rank(pct=True)
df["volume_rank"] = g["volume_ratio_5d"].rank(pct=True)
df["trend_rank"] = g["close_ma5_ratio"].rank(pct=True)
df["bb_rank"] = g["bb_position"].rank(pct=True)
df["vol_inv_rank"] = 1 - g["atr_ratio"].rank(pct=True)

# =========================================================
# ② featureを「全部0-1寄せ」に統一
# =========================================================
df["trend_feat"] = (
    0.5 * df["close_ma5_ratio"].clip(0.8, 1.2) +
    0.5 * df["close_ma25_ratio"].clip(0.8, 1.2)
)

df["momentum_feat"] = (
    0.6 * df["ret_rank"] +
    0.4 * df["volume_rank"]
)

df["break_feat"] = (
    0.5 * df["high_break_20d"] +
    0.5 * df["bb_rank"]
)

df["risk_feat"] = (
    0.5 * df["vol_inv_rank"] +
    0.5 * (1 - df["gap_up_ratio"].clip(0, 0.1))
)

# =========================================================
# ③ クロスセクション標準化（最重要バグ修正）
#    → これがないと“全員強い日”が発生する
# =========================================================
for col in ["trend_feat", "momentum_feat", "break_feat", "risk_feat"]:
    df[col] = g[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

# =========================================================
# ④ アルファスコア（安定化版）
# =========================================================
df["signal_score"] = (
    0.35 * df["trend_feat"] +
    0.30 * df["momentum_feat"] +
    0.20 * df["break_feat"] +
    0.15 * df["risk_feat"]
)

# =========================================================
# ⑤ スコアの再正規化（ここ重要）
#    → 勝率バグの最大原因はスコア暴走
# =========================================================
df["signal_score"] = g["signal_score"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-9)
)

# =========================================================
# ⑥ 強シグナル（絶対条件＋順位）
# =========================================================
df["signal_strong"] = (
    (df["signal_score"] > 1.0) &
    (df["ret_rank"] > 0.75) &
    (df["volume_rank"] > 0.75)
).astype(int)

# =========================================================
# ⑦ 弱シグナル（意味ある“準トレード”）
# =========================================================
df["signal_weak"] = (
    (df["signal_score"] > 0.3) &
    (df["signal_score"] <= 1.0) &
    (df["ret_rank"] > 0.5)
).astype(int)

# =========================================================
# ⑧ エントリー統合
# =========================================================
df["signal_entry"] = (
    df["signal_strong"] * 2 +
    df["signal_weak"] * 1
)

df["signal_trade"] = (df["signal_entry"] > 0).astype(int)

# =========================================================
# ⑨ デバッグ（重要）
# =========================================================
print("\n===== SIGNAL DISTRIBUTION =====")
print(df["signal_entry"].value_counts())

print("\n===== STRONG RATE =====")
print(df["signal_strong"].mean())

print("\n===== WEAK RATE =====")
print(df["signal_weak"].mean())

print("\n===== SCORE STATS =====")
print(df["signal_score"].describe())

# =========================================================
# save
# =========================================================
df.to_parquet(SAVE_PATH, index=False)

print("\nSaved:", SAVE_PATH)