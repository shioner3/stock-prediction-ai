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

INITIAL_CAPITAL = 1.0

THRESHOLD = 0.30
HOLD_DAYS = 7
STOP_LOSS = -0.03
TAKE_PROFIT = 0.10

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

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
# 🔥 バックテスト関数（共通）
# =========================
def run_backtest(train_df, test_df, label="TEST"):

    # 学習
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(train_df[FEATURES], train_df[TARGET])

    # 予測
    test_df = test_df.copy()
    test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]

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

            if ret < STOP_LOSS or ret > TAKE_PROFIT or d >= pos["exit_date"]:
                equity *= (1 + ret * pos["weight"])
            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー
        # =========================
        today_f = today.copy()

        today_f = today_f[today_f["pred"] > THRESHOLD]
        today_f = today_f[today_f["EMA_gap"] > 0]

        if today_f.empty:
            equity_curve.append(equity)
            continue

        # 🔥 レジーム
        market = today_f["Return_1"].mean()
        market_pred_mean = today_f["pred"].mean()

        if market < -0.01 or market_pred_mean < 0.30:
            weight_cap = 0.3
            top_n = 1
        else:
            weight_cap = 0.6
            top_n = 2

        # ハイブリッド
        today_f = make_hybrid_score(today_f)
        picks = today_f.sort_values("hybrid_score", ascending=False).head(top_n)

        total_pred = picks["pred"].sum()
        if total_pred == 0:
            equity_curve.append(equity)
            continue

        for _, row in picks.iterrows():

            if any(p["ticker"] == row["Ticker"] for p in positions):
                continue

            weight = min(row["pred"] / total_pred, weight_cap)

            positions.append({
                "ticker": row["Ticker"],
                "entry_price": row["Close"],
                "exit_date": d + pd.Timedelta(days=HOLD_DAYS),
                "weight": weight
            })

        equity_curve.append(equity)

    equity_curve = pd.Series(equity_curve)

    if len(equity_curve) < 2:
        return None

    returns = equity_curve.pct_change().dropna()

    return {
        "label": label,
        "CAGR": equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1,
        "Sharpe": returns.mean() / (returns.std() + 1e-9) * np.sqrt(252),
        "MaxDD": (equity_curve / equity_curve.cummax() - 1).min()
    }

# =========================
# 🔥 OOS
# =========================
OOS_START = 2024

train_df = df[df["Date"].dt.year < OOS_START]
test_df = df[df["Date"].dt.year >= OOS_START]

base = run_backtest(train_df, test_df, "BASE")

# =========================
# 🔥 ロバスト①（銘柄70%）
# =========================
tickers = test_df["Ticker"].unique()
sample = np.random.choice(tickers, int(len(tickers) * 0.7), replace=False)

test_70 = test_df[test_df["Ticker"].isin(sample)]

robust_1 = run_backtest(train_df, test_70, "Ticker70%")

# =========================
# 🔥 ロバスト②（期間80%）
# =========================
dates = sorted(test_df["Date"].unique())
cut = int(len(dates) * 0.8)

test_80 = test_df[test_df["Date"].isin(dates[:cut])]

robust_2 = run_backtest(train_df, test_80, "Time80%")

# =========================
# 結果
# =========================
results = pd.DataFrame([base, robust_1, robust_2])

print("\n=== OOS + ROBUST RESULT ===")
print(results)