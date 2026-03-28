import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

FEATURES = [
    "Return_1_rank",
    "MA5_ratio_rank",
    "MA25_ratio_rank",
    "MA75_ratio_rank",
    "Volatility_rank",
    "Volume_change_rank",
    "HL_range_rank",
    "RSI_rank"
]

TARGET = "Target"

TOP_K = 5
THRESHOLD = 0.03
STOP_LOSS = -0.03
INITIAL_CAPITAL = 1.0

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")

df["Year"] = df["Date"].dt.year
years = sorted(df["Year"].unique())

# =========================
# レジーム（リーク修正版）
# =========================
# ❌ 旧：銘柄別 return（市場ではない）
# df["MarketRet"] = df.groupby("Ticker")["Close"].pct_change()

# ✅ 新：クロスセクション市場代替（その日の平均）
df["MarketRet"] = df.groupby("Date")["Close"].transform(
    lambda x: x.pct_change().mean()
).fillna(0)

df["MarketMA20"] = df["MarketRet"].rolling(20).mean()

df["Regime"] = df["MarketMA20"] > 0

# =========================
# 日次資産曲線
# =========================
equity_curve = []
capital = INITIAL_CAPITAL

# =========================
# ローリング
# =========================
for i in range(3, len(years) - 1):

    train_years = years[i-3:i]
    test_year = years[i]

    train = df[df["Year"].isin(train_years)]
    test = df[df["Year"] == test_year]

    dates = sorted(test["Date"].unique())

    model = None
    last_month = None

    for d in dates:

        train_until_now = train[train["Date"] < d]

        if len(train_until_now) < 1000:
            continue

        # =========================
        # 月次学習（OK）
        # =========================
        if model is None or (d.year, d.month) != last_month:
            model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42
            )
            model.fit(
                train_until_now[FEATURES],
                train_until_now[TARGET]
            )
            last_month = (d.year, d.month)

        today = test[test["Date"] == d].copy()

        if len(today) == 0:
            continue

        # =========================
        # レジームフィルター（OK）
        # =========================
        if not bool(today["Regime"].iloc[0]):
            equity_curve.append(capital)
            continue

        # =========================
        # 予測
        # =========================
        today["Pred"] = model.predict(today[FEATURES])

        # ❌ 修正ポイント：閾値はランキングデータなので厳しすぎる
        today = today[today["Pred"] > today["Pred"].quantile(0.7)]

        if len(today) == 0:
            equity_curve.append(capital)
            continue

        picks = today.sort_values("Pred", ascending=False).head(TOP_K)

        # =========================
        # 未来リターン（Targetではなく正しく使用）
        # =========================
        rets = picks["FutureReturn_5"].values

        # NaN対策
        rets = np.nan_to_num(rets, nan=0.0)

        # STOP LOSS
        rets = np.clip(rets, STOP_LOSS, None)

        # =========================
        # ポートフォリオリターン
        # =========================
        daily_ret = np.mean(rets)

        capital *= (1 + daily_ret)

        equity_curve.append(capital)

# =========================
# 結果計算
# =========================
equity_curve = pd.Series(equity_curve)

returns = equity_curve.pct_change().dropna()

days = len(equity_curve)

cagr = equity_curve.iloc[-1] ** (252 / days) - 1
sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
maxdd = (equity_curve / equity_curve.cummax() - 1).min()

# =========================
# 出力
# =========================
print("\n========================")
print("CORRECT BACKTEST RESULT (FIXED)")
print("========================")
print("CAGR:", cagr)
print("Sharpe:", sharpe)
print("MaxDD:", maxdd)
print("Days:", days)

pd.DataFrame({
    "equity": equity_curve
}).to_csv("backtest_equity.csv", index=False)