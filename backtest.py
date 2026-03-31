import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from itertools import product

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

INITIAL_CAPITAL = 1.0

# 探索範囲
THRESHOLD_LIST = [0.28, 0.30, 0.32, 0.35]
HOLD_DAYS_LIST = [3, 5, 7]
TOP_N_LIST = [2, 3, 5]

# =========================
# データ読み込み
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
# 事前予測
# =========================
test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]

# =========================
# ハイブリッド
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
# バックテスト関数
# =========================
def run_fast_backtest(threshold, hold_days, top_n):

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

            if d >= pos["exit_date"]:
                cur = today[today["Ticker"] == pos["ticker"]]

                if not cur.empty:
                    price = cur["Close"].iloc[0]
                    ret = (price - pos["entry_price"]) / pos["entry_price"]

                    equity *= (1 + ret / top_n)
            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー
        # =========================
        today = today[today["pred"] > threshold]

        if today.empty:
            equity_curve.append(equity)
            continue

        today = make_hybrid_score(today)
        picks = today.sort_values("hybrid_score", ascending=False).head(top_n)

        for _, row in picks.iterrows():

            if any(p["ticker"] == row["Ticker"] for p in positions):
                continue

            positions.append({
                "ticker": row["Ticker"],
                "entry_price": row["Close"],
                "exit_date": d + pd.Timedelta(days=hold_days)
            })

        equity_curve.append(equity)

    equity_curve = pd.Series(equity_curve)

    if len(equity_curve) < 2:
        return None

    returns = equity_curve.pct_change().dropna()

    CAGR = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    Sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    MaxDD = (equity_curve / equity_curve.cummax() - 1).min()

    return CAGR, Sharpe, MaxDD

# =========================
# 最適化
# =========================
results = []

for th, hd, tn in product(THRESHOLD_LIST, HOLD_DAYS_LIST, TOP_N_LIST):

    print(f"Running: TH={th}, HOLD={hd}, TOP={tn}")

    res = run_fast_backtest(th, hd, tn)

    if res is None:
        continue

    CAGR, Sharpe, MaxDD = res

    results.append({
        "THRESHOLD": th,
        "HOLD_DAYS": hd,
        "TOP_N": tn,
        "CAGR": CAGR,
        "Sharpe": Sharpe,
        "MaxDD": MaxDD
    })

# =========================
# 結果
# =========================
result_df = pd.DataFrame(results)

print("\n=== BEST RESULT ===")
print(result_df.sort_values("CAGR", ascending=False).head(10))