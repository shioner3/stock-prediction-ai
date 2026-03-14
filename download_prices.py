import pandas as pd
import yfinance as yf
from tqdm import tqdm
import os

CSV_FILE = "data/data_j.csv"
PARQUET_FILE = "data/japan_stock.parquet"

# =========================
# 銘柄読み込み
# =========================

df_list = pd.read_csv(CSV_FILE, dtype=str)

df_list = df_list[
    df_list["市場・商品区分"].str.contains(
        "プライム|スタンダード|グロース",
        na=False
    )
]

tickers = df_list["コード"].tolist()
tickers = [t + ".T" for t in tickers]

print("銘柄数:", len(tickers))

# =========================
# 既存データ読み込み
# =========================

if os.path.exists(PARQUET_FILE):

    df_old = pd.read_parquet(PARQUET_FILE)

    last_date = df_old["Date"].max()

    print("既存データ最終日:", last_date)

else:

    df_old = pd.DataFrame()

    last_date = "2018-01-01"

# =========================
# 差分取得
# =========================

dfs = []

for ticker in tqdm(tickers):

    try:

        data = yf.download(
            ticker,
            start=last_date,
            progress=False
        )

        if data.empty:
            continue

        data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        data["Ticker"] = ticker

        data = data[
            ["Date","Open","High","Low","Close","Volume","Ticker"]
        ]

        dfs.append(data)

    except:
        continue

# =========================
# 結合
# =========================

if dfs:

    df_new = pd.concat(dfs, ignore_index=True)

    df = pd.concat([df_old, df_new], ignore_index=True)

    df = df.drop_duplicates(
        ["Date","Ticker"]
    ).sort_values(["Date","Ticker"])

else:

    df = df_old

# =========================
# 保存
# =========================

df.to_parquet(PARQUET_FILE)

print("保存完了:", df.shape)