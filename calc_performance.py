import pandas as pd
import os
import glob

# =========================
# 設定（バックテストと統一）
# =========================
PRED_LOG_PATTERN = "logs/predictions_*.csv"
PRICE_FILE = "stock_data/prices.parquet"

HOLD_DAYS = 7
STOP_LOSS = -0.02
TAKE_PROFIT = 0.08

# =========================
# 月別ログ
# =========================
today = pd.Timestamp.today().normalize()
month_str = today.strftime("%Y-%m")

PERF_LOG_MONTH = f"logs/performance_{month_str}.csv"

# =========================
# 初期チェック
# =========================
pred_files = glob.glob(PRED_LOG_PATTERN)

if len(pred_files) == 0:
    print("❌ predictionsファイルが存在しません")
    exit()

if not os.path.exists(PRICE_FILE):
    print("❌ prices.parquet が存在しません")
    exit()

# =========================
# 予測ログ
# =========================
df_list = []

for f in pred_files:
    try:
        df_tmp = pd.read_csv(f)
        df_list.append(df_tmp)
    except:
        print(f"読み込み失敗: {f}")

df_pred = pd.concat(df_list, ignore_index=True)

# =========================
# 価格データ
# =========================
df_price = pd.read_parquet(PRICE_FILE)

# =========================
# 列名調整
# =========================
if "コード" in df_pred.columns:
    df_pred = df_pred.rename(columns={"コード": "Ticker"})

df_price["Date"] = pd.to_datetime(df_price["Date"])
df_pred["predict_date"] = pd.to_datetime(df_pred["predict_date"])

# =========================
# 既存データ（月別）
# =========================
if os.path.exists(PERF_LOG_MONTH):
    df_perf_existing = pd.read_csv(PERF_LOG_MONTH)
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

    ticker = row["Ticker"]
    predict_date = row["predict_date"]

    key = f"{ticker}_{predict_date.strftime('%Y-%m-%d')}"
    if key in existing_keys:
        continue

    df_t = df_price[df_price["Ticker"] == ticker].sort_values("Date")

    # =========================
    # エントリー（翌日Open）
    # =========================
    entry_row = df_t[df_t["Date"] > predict_date].head(1)
    if entry_row.empty:
        continue

    entry_date = entry_row["Date"].iloc[0]
    entry_price = entry_row["Open"].iloc[0]

    # =========================
    # 保有期間
    # =========================
    future = df_t[df_t["Date"] >= entry_date].head(HOLD_DAYS)

    if len(future) < HOLD_DAYS:
        continue

    exit_price = None
    exit_date = None

    # =========================
    # 利確・損切り
    # =========================
    for _, r in future.iterrows():

        price = r["Close"]
        ret = (price - entry_price) / entry_price

        if ret <= STOP_LOSS or ret >= TAKE_PROFIT:
            exit_price = price
            exit_date = r["Date"]
            break

    # =========================
    # 期限決済
    # =========================
    if exit_price is None:
        exit_price = future["Close"].iloc[-1]
        exit_date = future["Date"].iloc[-1]

    ret = (exit_price / entry_price) - 1

    # =========================
    # レジーム
    # =========================
    market_score = row.get("Pred", 0)

    if market_score > 0.56:
        regime = "strong"
    elif market_score > 0.53:
        regime = "slightly_strong"
    elif market_score > 0.51:
        regime = "neutral"
    else:
        regime = "weak"

    results.append({
        "ticker": ticker,
        "predict_date": predict_date.strftime("%Y-%m-%d"),
        "entry_date": entry_date.strftime("%Y-%m-%d"),
        "exit_date": exit_date.strftime("%Y-%m-%d"),
        "return": ret,
        "win": int(ret > 0),
        "regime": regime
    })

# =========================
# 保存（月別のみ）
# =========================
if results:
    df_new = pd.DataFrame(results)

    if os.path.exists(PERF_LOG_MONTH):
        df_month = pd.concat([df_perf_existing, df_new], ignore_index=True)
    else:
        df_month = df_new

    df_month.to_csv(PERF_LOG_MONTH, index=False)

    print("✅ 実績追加:", len(df_new))
else:
    print("追加データなし")

# =========================
# 指標（月別のみ）
# =========================
if os.path.exists(PERF_LOG_MONTH):

    df = pd.read_csv(PERF_LOG_MONTH)

    print(f"\n📊 今月実績 ({month_str})")

    win_rate = df["win"].mean()
    avg_return = df["return"].mean()
    std = df["return"].std()
    sharpe = avg_return / std if std != 0 else 0

    print("勝率:", round(win_rate, 3))
    print("平均リターン:", round(avg_return, 4))
    print("Sharpe:", round(sharpe, 3))

    # 市場別
    print("\n📊 市場別実績")

    for regime in ["strong", "slightly_strong", "neutral", "weak"]:

        df_r = df[df["regime"] == regime]

        if len(df_r) < 5:
            print(f"{regime}: データ不足")
            continue

        win_rate = df_r["win"].mean()
        avg_return = df_r["return"].mean()
        std = df_r["return"].std()
        sharpe = avg_return / std if std != 0 else 0

        print(f"\n[{regime}]")
        print("勝率:", round(win_rate, 3))
        print("平均リターン:", round(avg_return, 4))
        print("Sharpe:", round(sharpe, 3))