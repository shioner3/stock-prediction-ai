import pandas as pd
import os
from datetime import datetime

# =========================
# 設定
# =========================
PRED_LOG = "logs/predictions.csv"
PERF_LOG = "logs/performance.csv"
PRICE_FILE = "stock_data/prices.parquet"

# =========================
# 初期チェック
# =========================
if not os.path.exists(PRED_LOG):
    print("❌ predictions.csv が存在しません")
    exit()

if not os.path.exists(PRICE_FILE):
    print("❌ prices.parquet が存在しません")
    exit()

# =========================
# 読み込み
# =========================
df_pred = pd.read_csv(PRED_LOG)
df_price = pd.read_parquet(PRICE_FILE)

df_price["Date"] = pd.to_datetime(df_price["Date"])
df_pred["predict_date"] = pd.to_datetime(df_pred["predict_date"])
df_pred["target_date"] = pd.to_datetime(df_pred["target_date"])

today = pd.Timestamp.today().normalize()

# =========================
# 既存パフォーマンス読み込み（重複防止）
# =========================
if os.path.exists(PERF_LOG):
    df_perf_existing = pd.read_csv(PERF_LOG)
else:
    df_perf_existing = pd.DataFrame()

existing_keys = set()

if not df_perf_existing.empty:
    existing_keys = set(
        df_perf_existing["ticker"] + "_" + df_perf_existing["predict_date"]
    )

# =========================
# 実績計算
# =========================
results = []

for _, row in df_pred.iterrows():

    ticker = row["ticker"]
    predict_date = row["predict_date"]
    target_date = row["target_date"]

    key = f"{ticker}_{predict_date}"

    # すでに計算済みならスキップ
    if key in existing_keys:
        continue

    # まだ未来ならスキップ
    if target_date > today:
        continue

    df_t = df_price[df_price["Ticker"] == ticker]

    start_price = df_t[df_t["Date"] >= predict_date]["Close"].head(1)
    end_price = df_t[df_t["Date"] >= target_date]["Close"].head(1)

    if start_price.empty or end_price.empty:
        continue

    ret = (end_price.values[0] / start_price.values[0]) - 1

    results.append({
        "ticker": ticker,
        "predict_date": predict_date.strftime("%Y-%m-%d"),
        "target_date": target_date.strftime("%Y-%m-%d"),
        "return": ret,
        "win": int(ret > 0)
    })

# =========================
# 保存
# =========================
if results:
    df_new = pd.DataFrame(results)

    if os.path.exists(PERF_LOG):
        df_all = pd.concat([df_perf_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(PERF_LOG, index=False)

    print("✅ 実績追加:", len(df_new))
else:
    print("追加データなし")

# =========================
# 指標計算
# =========================
if os.path.exists(PERF_LOG):
    df = pd.read_csv(PERF_LOG)

    win_rate = df["win"].mean()
    avg_return = df["return"].mean()
    std = df["return"].std()

    sharpe = avg_return / std if std != 0 else 0

    print("\n📊 実績サマリー")
    print("勝率:", round(win_rate, 3))
    print("平均リターン:", round(avg_return, 4))
    print("Sharpe:", round(sharpe, 3))