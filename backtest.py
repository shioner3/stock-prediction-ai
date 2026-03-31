import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

FEATURES = [
    "Return_1","Return_3","Return_5",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio","Volume_accel",
    "HL_range",
    "EMA_gap",
    "Momentum_5","Momentum_10",
    "ATR_ratio",
    "RSI"
]

TARGET = "Target"

TOP_N = 3
HOLD_DAYS = 3
THRESHOLD = 0.35
INITIAL_CAPITAL = 1.0

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# 学習（1回だけ）
# =========================
train_df = df[df["Date"].dt.year < 2024]
test_df = df[df["Date"].dt.year >= 2024].copy()

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

model.fit(train_df[FEATURES], train_df[TARGET])

# =========================
# 🔥 事前予測（高速化の核）
# =========================
test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]

# =========================
# ハイブリッドスコア
# =========================
def make_hybrid_score(df):
    df = df.copy()

    df["mom_rank"] = df["Return_5"].rank(pct=True)
    df["trend_rank"] = df["EMA_gap"].rank(pct=True)
    df["vol_rank"] = (-df["Volatility"]).rank(pct=True)

    df["hybrid_score"] = (
        0.4 * df["mom_rank"] +
        0.3 * df["trend_rank"] +
        0.3 * df["vol_rank"]
    )

    return df

# =========================
# 🔥 バックテスト
# =========================
equity = INITIAL_CAPITAL
equity_curve = []

positions = []  # 保有ポジション

dates = sorted(test_df["Date"].unique())

for d in dates:

    today = test_df[test_df["Date"] == d]

    # =========================
    # 決済処理
    # =========================
    new_positions = []

    for pos in positions:

        if d >= pos["exit_date"]:
            cur = today[today["Ticker"] == pos["ticker"]]

            if not cur.empty:
                price = cur["Close"].iloc[0]
                ret = (price - pos["entry_price"]) / pos["entry_price"]

                # 🔥 等ウェイト
                equity *= (1 + ret / TOP_N)
        else:
            new_positions.append(pos)

    positions = new_positions

    # =========================
    # エントリー候補
    # =========================
    today = today.copy()

    # フィルター
    today = today[today["pred"] > THRESHOLD]

    if today.empty:
        equity_curve.append(equity)
        continue

    # ハイブリッドスコア
    today = make_hybrid_score(today)

    picks = today.sort_values("hybrid_score", ascending=False).head(TOP_N)

    # =========================
    # エントリー
    # =========================
    for _, row in picks.iterrows():

        # 重複回避
        if any(p["ticker"] == row["Ticker"] for p in positions):
            continue

        entry_price = row["Close"]

        if entry_price <= 0:
            continue

        positions.append({
            "ticker": row["Ticker"],
            "entry_price": entry_price,
            "exit_date": d + pd.Timedelta(days=HOLD_DAYS)
        })

    equity_curve.append(equity)

# =========================
# 結果
# =========================
equity_curve = pd.Series(equity_curve)
returns = equity_curve.pct_change().dropna()

print("\n=== FAST BACKTEST (FIXED) ===")
print("CAGR:", equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1)
print("Sharpe:", returns.mean() / (returns.std() + 1e-9) * np.sqrt(252))
print("MaxDD:", (equity_curve / equity_curve.cummax() - 1).min())