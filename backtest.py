import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# =========================
# 設定（固定）
# =========================
DATA_PATH = "ml_dataset.parquet"

FEATURES = [
    "Return_1_rank","Return_3_rank","MA3_ratio_rank","MA5_ratio_rank",
    "MA10_ratio_rank","Volatility_rank","Volume_change_rank",
    "Volume_ratio_rank","HL_range_rank","RSI_rank"
]

TARGET = "Target"

STOP_LOSS = -0.03
INITIAL_CAPITAL = 1.0
MAX_WEIGHT = 0.4

PRED_THRESHOLD = 0.51
MARKET_THRESHOLD = 0.48

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# 市場レジーム
# =========================
df["MarketRet"] = df.groupby("Date")["Close"].transform("mean").pct_change().fillna(0)

market = df.groupby("Date")["MarketRet"].mean().sort_index()
market_ma20 = market.rolling(20).mean()

df = df.merge(
    pd.DataFrame({"MarketMA20": market_ma20}).fillna(0),
    left_on="Date",
    right_index=True,
    how="left"
)

df["Regime"] = np.where(
    df["MarketMA20"] > 0.002, "bull",
    np.where(df["MarketMA20"] > -0.002, "neutral", "bear")
)

# =========================
# 🔥 完全アウトオブサンプル
# =========================
OOS_START = 2024

train_df_full = df[df["Date"].dt.year < OOS_START]
test_df_oos = df[df["Date"].dt.year >= OOS_START]

print("\n=== OOS TEST ===")
print("Train:", train_df_full["Date"].min(), "~", train_df_full["Date"].max())
print("Test :", test_df_oos["Date"].min(), "~", test_df_oos["Date"].max())

# =========================
# 🔥 ロバスト性用関数
# =========================
def run_backtest(test_df, train_df, label="BASE"):

    equity = INITIAL_CAPITAL
    equity_curve = []

    model = None
    positions = []

    trade_returns = []
    trade_count = 0

    dates = sorted(test_df["Date"].unique())
    prev_month = None

    for j in range(len(dates) - 1):

        d = dates[j]
        next_d = dates[j + 1]

        today = test_df[test_df["Date"] == d].copy()
        tomorrow = test_df[test_df["Date"] == next_d].copy()

        if today.empty:
            equity_curve.append(equity)
            continue

        regime = today["Regime"].iloc[0]

        # =========================
        # 月次学習
        # =========================
        current_month = d.month

        if model is None or current_month != prev_month:

            train_until = train_df[train_df["Date"] < d]

            if len(train_until) > 2000:
                model = LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    random_state=42
                )
                model.fit(train_until[FEATURES], train_until[TARGET])

            prev_month = current_month

        if model is None:
            equity_curve.append(equity)
            continue

        # =========================
        # 予測
        # =========================
        today_pred = today.copy()
        today_pred["pred"] = model.predict(today_pred[FEATURES])

        # 地雷フィルタ
            
        if "limit_up_flag" in today_pred.columns:
            today_pred = today_pred[today_pred["limit_up_flag"] == 0]
            
            # 出来高フィルター（追加）
        if "Volume" in today_pred.columns:
            today_pred = today_pred[today_pred["Volume"] > 10000]

        if today_pred.empty:
            equity_curve.append(equity)
            continue

        # 市場
        market_score = today_pred["pred"].mean()
        if np.isnan(market_score):
            market_score = 0

        if market_score < MARKET_THRESHOLD:
            max_positions = 2
        else:
            max_positions = 5 if regime == "bull" else 3

        # スコア
        candidates = today_pred[today_pred["pred"] > PRED_THRESHOLD]

        if len(candidates) < 3:
            candidates = today_pred.sort_values("pred", ascending=False).head(5)

        # エントリー
        if len(positions) == 0:

            picks = candidates.sort_values("pred", ascending=False).head(max_positions)

            picks["vol"] = picks["Volatility_rank"] + 1e-6
            picks["weight"] = 1 / np.sqrt(picks["vol"])
            picks["weight"] /= picks["weight"].sum()

            hold_days = 3 if regime != "bull" else 4

            for _, row in picks.iterrows():

                tmr = tomorrow[tomorrow["Ticker"] == row["Ticker"]]
                if tmr.empty:
                    continue

                positions.append({
                    "ticker": row["Ticker"],
                    "entry_price": tmr["Open"].iloc[0],
                    "entry_day": j,
                    "weight": row["weight"],
                    "hold_days": hold_days
                })

                trade_count += 1

        # 管理
        new_positions = []

        for pos in positions:

            cur = today[today["Ticker"] == pos["ticker"]]

            if cur.empty:
                new_positions.append(pos)
                continue

            price = cur["Close"].iloc[0]
            ret = (price - pos["entry_price"]) / pos["entry_price"]

            hold_days_now = j - pos["entry_day"] + 1

            if ret <= STOP_LOSS or hold_days_now >= pos["hold_days"]:

                pnl = ret
                equity *= (1 + pnl * pos["weight"])
                trade_returns.append(pnl)
                continue

            new_positions.append(pos)

        positions = new_positions
        equity_curve.append(equity)

    equity_curve = pd.Series(equity_curve)

    if len(equity_curve) < 2:
        return None

    returns = equity_curve.pct_change().dropna()

    return {
        "label": label,
        "CAGR": equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1,
        "Sharpe": returns.mean() / (returns.std() + 1e-9) * np.sqrt(252),
        "MaxDD": (equity_curve / equity_curve.cummax() - 1).min(),
        "Trades": trade_count
    }

# =========================
# 🔥 ベース
# =========================
base_result = run_backtest(test_df_oos, train_df_full, "BASE")

# =========================
# 🔥 ロバスト性チェック
# =========================

# ① 銘柄削減
tickers = test_df_oos["Ticker"].unique()
reduced = np.random.choice(tickers, int(len(tickers)*0.7), replace=False)
test_reduced = test_df_oos[test_df_oos["Ticker"].isin(reduced)]

robust_1 = run_backtest(test_reduced, train_df_full, "Ticker70%")

# ② 期間削減
dates = sorted(test_df_oos["Date"].unique())
cut = int(len(dates)*0.8)
test_short = test_df_oos[test_df_oos["Date"].isin(dates[:cut])]

robust_2 = run_backtest(test_short, train_df_full, "Time80%")

# =========================
# 出力
# =========================
results = pd.DataFrame([base_result, robust_1, robust_2])

print("\n=== ROBUST TEST ===")
print(results)