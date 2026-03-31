import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# =========================
# 設定（固定）
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

INITIAL_CAPITAL = 1.0

# 🔥 固定パラメータ
THRESHOLD = 0.30
HOLD_DAYS = 7
TOP_N = 2
STOP_LOSS = -0.05

# =========================
# データ読み込み
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# 学習（1回）
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
# 予測
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
        0.5 * df["mom_rank"] +
        0.3 * df["trend_rank"] +
        0.2 * df["vol_rank"]
    )

    return df

# =========================
# バックテスト
# =========================
equity = INITIAL_CAPITAL
positions = []
equity_curve = []

dates = sorted(test_df["Date"].unique())

for d in dates:

    today = test_df[test_df["Date"] == d]

    # =========================
    # 決済
    # =========================
    new_positions = []

    for pos in positions:

        cur = today[today["Ticker"] == pos["ticker"]]

        if cur.empty:
            new_positions.append(pos)
            continue

        price = cur["Close"].iloc[0]
        ret = (price - pos["entry_price"]) / pos["entry_price"]

        # 🔥 損切り
        if ret < STOP_LOSS:
            equity *= (1 + ret * pos["weight"])
            continue

        # 通常決済
        if d >= pos["exit_date"]:
            equity *= (1 + ret * pos["weight"])
            continue

        new_positions.append(pos)

    positions = new_positions

    # =========================
    # エントリー候補
    # =========================
    today = today.copy()

    # 🔥 ① 確率フィルター
    today = today[today["pred"] > THRESHOLD]

    # 🔥 ② トレンドフィルター
    today = today[today["EMA_gap"] > 0]

    if today.empty:
        equity_curve.append(equity)
        continue

    # 🔥 ③ ハイブリッド
    today = make_hybrid_score(today)

    picks = today.sort_values("hybrid_score", ascending=False).head(TOP_N)

    # =========================
    # 🔥 ロット調整（predベース）
    # =========================
    total_pred = picks["pred"].sum()

    if total_pred == 0:
        equity_curve.append(equity)
        continue

    # =========================
    # エントリー
    # =========================
    for _, row in picks.iterrows():

        if any(p["ticker"] == row["Ticker"] for p in positions):
            continue

        weight = row["pred"] / total_pred  # 🔥 ここがロット調整

        positions.append({
            "ticker": row["Ticker"],
            "entry_price": row["Close"],
            "exit_date": d + pd.Timedelta(days=HOLD_DAYS),
            "weight": weight
        })

    equity_curve.append(equity)

# =========================
# 結果
# =========================
equity_curve = pd.Series(equity_curve)

returns = equity_curve.pct_change().dropna()

print("\n=== FINAL BACKTEST (HYBRID) ===")
print("CAGR:", equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1)
print("Sharpe:", returns.mean() / (returns.std() + 1e-9) * np.sqrt(252))
print("MaxDD:", (equity_curve / equity_curve.cummax() - 1).min())