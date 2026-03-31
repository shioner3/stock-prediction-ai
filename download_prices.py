import pandas as pd
import yfinance as yf
from tqdm import tqdm
import os
import time

# =========================
# ファイル
# =========================
CSV_FILE = "data_j.csv"
PARQUET_FILE = "stock_data/prices.parquet"


RECENT_DAYS = 10
MIN_COUNT = 3000

BATCH_SIZE = 50
SLEEP_TIME = 1
RETRY = 3

SAFE_BUFFER_DAYS = 7

os.makedirs("stock_data", exist_ok=True)


# =========================
# 正規化
# =========================
def normalize_ticker(t):
    t = str(t).strip().upper()
    if not t.endswith(".T"):
        t += ".T"
    return t


# =========================
# 銘柄マスタ
# =========================
df_list = pd.read_csv(CSV_FILE, dtype=str)

df_list = df_list[
    df_list["市場・商品区分"].str.contains("プライム|スタンダード|グロース", na=False)
]

df_list["Ticker"] = df_list["コード"].apply(normalize_ticker)
df_list["Name"] = df_list["銘柄名"].astype(str).str.strip()

tickers = df_list["Ticker"].tolist()
name_dict = dict(zip(df_list["Ticker"], df_list["Name"]))

print("銘柄数:", len(tickers))

today = pd.Timestamp.now().normalize()


# =========================
# 株価データ読み込み
# =========================
if os.path.exists(PARQUET_FILE):
    df_existing = pd.read_parquet(PARQUET_FILE)

    df_existing["Ticker"] = df_existing["Ticker"].apply(normalize_ticker)
    df_existing["Date"] = pd.to_datetime(df_existing["Date"]).dt.tz_localize(None)

    df_existing = df_existing.dropna(subset=["Date"])
    df_existing = df_existing[df_existing["Date"] <= today]

else:
    df_existing = pd.DataFrame(
        columns=["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Name"]
    )


# =========================
# 最終取得日
# =========================
last_dates = {}

if not df_existing.empty:
    last_dates = df_existing.groupby("Ticker")["Date"].max().to_dict()


# =========================
# yfinance取得
# =========================
def fetch_batch(batch, start=None):
    for attempt in range(RETRY):
        try:
            data = yf.download(
                batch,
                start=start,
                group_by="ticker",
                auto_adjust=True,
                threads=False,
                progress=False
            )
            return data
        except Exception as e:
            print(f"Retry {attempt+1}: {e}")
            time.sleep(2)
    return None


# =========================
# 🔥 株価差分取得
# =========================
targets = {}

for t in tickers:
    if t in last_dates:
        start = last_dates[t] - pd.Timedelta(days=SAFE_BUFFER_DAYS)
    else:
        start = pd.Timestamp("2018-01-01")

    targets[t] = start


print("\n=== 差分取得 ===")

dfs_new = []

for i in tqdm(range(0, len(tickers), BATCH_SIZE)):

    batch = tickers[i:i+BATCH_SIZE]

    start = min([targets[t] for t in batch])
    start_str = pd.to_datetime(start).strftime("%Y-%m-%d")

    data = fetch_batch(batch, start=start_str)

    if data is None:
        continue

    for ticker in batch:
        try:
            if ticker not in data:
                continue

            df_t = data[ticker].copy()
            if df_t.empty:
                continue

            df_t = df_t.reset_index()
            df_t["Date"] = pd.to_datetime(df_t["Date"]).dt.tz_localize(None)

            df_t["Ticker"] = ticker
            df_t["Name"] = name_dict.get(ticker)

            df_t = df_t[
                ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Name"]
            ]

            dfs_new.append(df_t)

        except:
            continue

    time.sleep(SLEEP_TIME)


# =========================
# 結合
# =========================
df_new = pd.concat(dfs_new, ignore_index=True) if dfs_new else pd.DataFrame()

df_all = pd.concat([df_existing, df_new], ignore_index=True)

df_all = df_all.drop_duplicates(subset=["Date", "Ticker"], keep="last")


# =========================
# 完全性チェック
# =========================
counts = df_all["Date"].value_counts()

print("\n=== 最新付近チェック ===")
print(counts.sort_index().tail(10))

valid_dates = counts[counts >= MIN_COUNT].index

latest_valid = max(valid_dates) if len(valid_dates) > 0 else None

print("\n=== 判定 ===")
print("最新日:", df_all["Date"].max())
print("有効最新日:", latest_valid)
print("銘柄数:", counts.get(latest_valid, 0))


# =========================
# 保存（株価）
# =========================
df_all.to_parquet(PARQUET_FILE, index=False)


# =========================================================
# 🔥 決算データ（月1キャッシュ）
# =========================================================
def fetch_earnings(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.earnings_dates
        if df is None:
            return None

        df = df.reset_index()
        df["Ticker"] = ticker
        return df

    except:
        return None


def load_or_update_earnings():
    today_month = pd.Timestamp.now().strftime("%Y-%m")

    # 既存あり
    if os.path.exists(EARNINGS_FILE) and os.path.exists(EARNINGS_META):
        meta = pd.read_pickle(EARNINGS_META)

        if meta.get("month") == today_month:
            print("\n⚡ earnings: キャッシュ使用")
            return pd.read_parquet(EARNINGS_FILE)

    # 更新
    print("\n🔄 earnings: 月次フル更新")

    dfs = []

    for t in tqdm(tickers[:]):  
        df = fetch_earnings(t)
        if df is not None:
            dfs.append(df)

        time.sleep(0.2)

    if dfs:
        result = pd.concat(dfs, ignore_index=True)
    else:
        result = pd.DataFrame()

    result.to_parquet(EARNINGS_FILE, index=False)
    pd.to_pickle({"month": today_month}, EARNINGS_META)

    return result


# =========================
# 実行（決算更新）
# =========================
earnings_df = load_or_update_earnings()

print("\n決算データ件数:", len(earnings_df))


# =========================
# 完了
# =========================
print("\n=== 完了 ===")
print("総株価行数:", len(df_all))