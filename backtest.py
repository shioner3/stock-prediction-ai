import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

HOLD_DAYS = 7
INITIAL_CAPITAL = 1.0
MAX_POSITIONS = 5

TOP_RATE = 0.05
ALPHA = 2.0

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date","Ticker"])

FEATURES = [c for c in df.columns if c not in ["Date","Ticker","Name","Target"]]
years = sorted(df["Date"].dt.year.unique())

# =========================
# モデル
# =========================
def train_model(train_df):
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train_df[FEATURES], train_df["Target"])
    return model

# =========================
# バックテスト
# =========================
def run_backtest(train_df, test_df):

    model = train_model(train_df)

    test_df = test_df.copy()
    test_df["score"] = model.predict(test_df[FEATURES])

    # 日付ごとに高速アクセス
    grouped = {d: g for d, g in test_df.groupby("Date")}
    dates = sorted(grouped.keys())
    date_index = {d:i for i,d in enumerate(dates)}

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []

    trade_count = 0

    for d in dates:

        today = grouped[d]
        daily_pnl = 0

        # =========================
        # 🔥 決済（exit日のみ）
        # =========================
        new_positions = []

        for pos in positions:

            # exit日になったら決済
            if d == pos["exit_date"]:

                cur = today[today["Ticker"] == pos["ticker"]]

                if cur.empty:
                    continue

                # 🔥 exitはOpenで約定
                exit_price = cur["Open"].iloc[0]

                ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                daily_pnl += pos["capital"] * ret

            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー
        # =========================
        if date_index[d] + 1 < len(dates):

            available = MAX_POSITIONS - len(positions)
            if available <= 0:
                equity += daily_pnl
                equity_curve.append(equity)
                continue

            next_day = dates[date_index[d]+1]
            next_data = grouped[next_day]

            # フィルタ
            today_f = today[today["Trend_5_z"] > -0.5]

            if len(today_f) == 0:
                equity += daily_pnl
                equity_curve.append(equity)
                continue

            # 上位抽出
            TOP_K = max(3, int(len(today_f) * TOP_RATE))
            picks = today_f.nlargest(TOP_K, "score")

            picks = picks.head(available)

            if len(picks) > 0:

                free_cash = equity - sum(p["capital"] for p in positions)

                scores = picks["score"].values
                scores = scores - scores.mean()
                scores = np.clip(scores, 0, None)

                if scores.sum() == 0:
                    equity += daily_pnl
                    equity_curve.append(equity)
                    continue

                weights = scores ** ALPHA
                weights = weights / weights.sum()

                tickers = picks["Ticker"].values

                for i in range(len(picks)):

                    ticker = tickers[i]

                    if any(p["ticker"] == ticker for p in positions):
                        continue

                    next_row = next_data[next_data["Ticker"] == ticker]
                    if next_row.empty:
                        continue

                    capital = free_cash * weights[i]
                    capital = min(capital, equity * 0.3)

                    if capital <= 0:
                        continue

                    entry_price = next_row["Open"].iloc[0]

                    positions.append({
                        "ticker": ticker,
                        "entry_price": entry_price,
                        "exit_date": next_day + pd.Timedelta(days=HOLD_DAYS),
                        "capital": capital
                    })

                    trade_count += 1

        # =========================
        # 更新
        # =========================
        equity += daily_pnl
        equity_curve.append(equity)

    equity_curve = pd.Series(equity_curve)
    returns = equity_curve.pct_change().dropna()

    CAGR = equity_curve.iloc[-1] ** (252/len(equity_curve)) - 1
    Sharpe = returns.mean()/(returns.std()+1e-9)*np.sqrt(252)
    MaxDD = (equity_curve/equity_curve.cummax()-1).min()

    return CAGR, Sharpe, MaxDD, trade_count


# =========================
# 実行
# =========================
results = []

for y in years:

    if y < 2022:
        continue

    train_df = df[df["Date"].dt.year < y]
    test_df = df[df["Date"].dt.year == y]

    CAGR, Sharpe, MaxDD, trades = run_backtest(train_df, test_df)

    print(f"{y} CAGR:{CAGR:.3f} Sharpe:{Sharpe:.2f}")

    results.append([y, CAGR, Sharpe, MaxDD, trades])

print(pd.DataFrame(results, columns=["year","CAGR","Sharpe","MaxDD","Trades"]))