import pandas as pd
import yfinance as yf
from tqdm import tqdm
import os
import time

CSV_FILE = "data_j.csv"
PARQUET_FILE = "stock_data/prices.parquet"

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
    df_list["市場・商品区分"].str.contains(
        "プライム|スタンダード|グロース", na=False
    )
]

df_list["Ticker"] = df_list["コード"].apply(normalize_ticker)
df_list["Name"] = df_list["銘柄名"].astype(str).str.strip()

tickers = df_list["Ticker"].tolist()
name_dict = dict(zip(df_list["Ticker"], df_list["Name"]))

print("銘柄数:", len(tickers))

os.makedirs("stock_data", exist_ok=True)

today = pd.Timestamp.now().normalize()

# =========================
# 既存データ
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
# 最新日
# =========================
if not df_existing.empty:

    last_dates_df = (
        df_existing.groupby("Ticker")["Date"]
        .max()
        .reset_index()
        .rename(columns={"Date": "last_date"})
    )

    last_dates_dict = dict(zip(last_dates_df["Ticker"], last_dates_df["last_date"]))
    global_latest = df_existing["Date"].max()

else:
    last_dates_dict = {}
    global_latest = None

# =========================
# ① 差分取得
# =========================
dfs = []
api_calls = 0

for ticker in tqdm(tickers):

    last_date = last_dates_dict.get(ticker)

    if last_date is not None:
        last_date = pd.to_datetime(last_date)

        if global_latest is not None:
            if last_date >= global_latest - pd.Timedelta(days=1):
                continue

        start_dt = last_date + pd.Timedelta(days=1)

    else:
        start_dt = pd.Timestamp("2018-01-01")

    try:
        api_calls += 1

        data = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), progress=False)

        if data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)

        data["Ticker"] = ticker
        data["Name"] = name_dict.get(ticker)

        data = data[
            ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Name"]
        ]

        dfs.append(data)

        time.sleep(0.05)

    except Exception as e:
        print(f"Error: {ticker}", e)

# =========================
# 一旦保存
# =========================
if dfs:
    df_new = pd.concat(dfs, ignore_index=True)
    df_existing = pd.concat([df_existing, df_new], ignore_index=True)

    df_existing = df_existing.drop_duplicates(subset=["Date", "Ticker"])

# =========================
# ② 欠損検出
# =========================
latest = df_existing["Date"].max()

latest_tickers = df_existing[df_existing["Date"] == latest]["Ticker"]

missing_tickers = list(set(tickers) - set(latest_tickers))

print("\n=== 欠損チェック ===")
print("最新日:", latest)
print("欠損数:", len(missing_tickers))

# =========================
# ③ 欠損再取得
# =========================
dfs_retry = []

for ticker in tqdm(missing_tickers):

    try:
        api_calls += 1

        data = yf.download(
            ticker,
            start=(latest - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            progress=False
        )

        if data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)

        data["Ticker"] = ticker
        data["Name"] = name_dict.get(ticker)

        data = data[
            ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Name"]
        ]

        dfs_retry.append(data)

        time.sleep(0.05)

    except Exception as e:
        print(f"Retry Error: {ticker}", e)

# =========================
# 最終保存
# =========================
if dfs_retry:
    df_retry = pd.concat(dfs_retry, ignore_index=True)
    df_existing = pd.concat([df_existing, df_retry], ignore_index=True)

df_existing = df_existing.drop_duplicates(subset=["Date", "Ticker"])
df_existing.to_parquet(PARQUET_FILE, index=False)

# =========================
# 結果
# =========================
print("\n=== 完了 ===")
print("API回数:", api_calls)
print("最終行数:", len(df_existing))
print("最新日:", df_existing["Date"].max())
print("最新日の銘柄数:", df_existing[df_existing["Date"] == df_existing["Date"].max()]["Ticker"].nunique())