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

# =========================
# データ読み込み
# =========================
df = pd.read_parquet(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")

df["Year"] = df["Date"].dt.year
years = sorted(df["Year"].unique())

# =========================
# 市場レジーム（リーク修正版）
# → Target禁止、過去リターンのみ
# =========================
df["MarketRet"] = df.groupby("Date")["Close"].pct_change()
df["MarketRet"] = df["MarketRet"].fillna(0)

df["MarketMA20"] = df.groupby("Ticker")["MarketRet"].transform(
    lambda x: x.rolling(20).mean()
)

df["Regime"] = df["MarketMA20"] > 0


# =========================
# 取引ログ
# =========================
all_returns = []

# =========================
# ローリング検証
# =========================
for i in range(3, len(years) - 1):

    train_years = years[i-3:i]
    test_year = years[i]

    print(f"\n===== {train_years} → {test_year} =====")

    train = df[df["Year"].isin(train_years)]
    test_df = df[df["Year"] == test_year]

    dates = sorted(test_df["Date"].unique())

    model = None
    last_month = None

    for d in dates:

        train_until_now = train[train["Date"] < d]

        if len(train_until_now) < 1000:
            continue

        # =========================
        # 月次再学習
        # =========================
        current_month = (d.year, d.month)

        if model is None or current_month != last_month:
            model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42
            )
            model.fit(train_until_now[FEATURES], train_until_now[TARGET])
            last_month = current_month

        # =========================
        # 当日データ
        # =========================
        today = test_df[test_df["Date"] == d].copy()

        if len(today) == 0:
            continue

        # =========================
        # レジーム（未来禁止）
        # =========================
        regime_ok = today["Regime"].iloc[0]
        if not regime_ok:
            continue

        # =========================
        # 予測
        # =========================
        today["Pred"] = model.predict(today[FEATURES])

        # =========================
        # フィルタ
        # =========================
        today = today[today["Pred"] > THRESHOLD]

        if len(today) == 0:
            continue

        # =========================
        # TOP K
        # =========================
        picks = today.sort_values("Pred", ascending=False).head(TOP_K)

        # =========================
        # 銘柄単位リターン
        # =========================
        rets = picks["FutureReturn_5"].values

        # STOP LOSS（銘柄単位）
        rets = np.clip(rets, STOP_LOSS, None)

        all_returns.extend(rets)


# =========================
# 集計
# =========================
res = pd.Series(all_returns)

if len(res) == 0:
    print("No trades")
    exit()

# 資金曲線
equity = (1 + res).cumprod()

cagr = equity.iloc[-1] ** (252 / len(res)) - 1
sharpe = res.mean() / (res.std() + 1e-9) * np.sqrt(252)
maxdd = (equity / equity.cummax() - 1).min()

# =========================
# 結果
# =========================
print("\n========================")
print("CLEAN BACKTEST RESULT")
print("========================")
print("CAGR:", cagr)
print("Sharpe:", sharpe)
print("MaxDD:", maxdd)
print("Trades:", len(res))

# 保存
pd.DataFrame({
    "return": res
}).to_csv("backtest_clean.csv", index=False)