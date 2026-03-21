import pandas as pd
import yfinance as yf
from tqdm import tqdm
import duckdb
import os

CSV_FILE = "data_j.csv"
PARQUET_FILE = "stock_data/prices.parquet"

# =========================
# 銘柄読み込み
# =========================
df_list = pd.read_csv(CSV_FILE, dtype=str)

df_list = df_list[
    df_list["市場・商品区分"].str.contains(
        "プライム|スタンダード|グロース", na=False
    )
]

tickers = df_list["コード"].tolist()
tickers = [t + ".T" for t in tickers]

print("銘柄数:", len(tickers))

# =========================
# データフォルダ作成
# =========================
os.makedirs("stock_data", exist_ok=True)

# =========================
# DuckDB接続（SQL用）
# =========================
con = duckdb.connect()

# =========================
# 既存Parquet読み込み
# =========================
if os.path.exists(PARQUET_FILE):

    df_existing = pd.read_parquet(PARQUET_FILE)

else:

    df_existing = pd.DataFrame(
        columns=["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
    )

# =========================
# tickerごとの最新日取得
# =========================
if not df_existing.empty:

    last_dates_df = con.execute(
        f"""
        SELECT Ticker, MAX(Date) AS last_date
        FROM '{PARQUET_FILE}'
        GROUP BY Ticker
        """
    ).fetchdf()

else:

    last_dates_df = pd.DataFrame(columns=["Ticker", "last_date"])

last_dates_dict = dict(zip(last_dates_df["Ticker"], last_dates_df["last_date"]))

print("Parquetに保存されている各Tickerの最新日:")
print(last_dates_df.head())

print("取得対象のtickers例:")
print(tickers[:5])

# =========================
# 差分取得
# =========================
dfs = []

today = pd.Timestamp.today().normalize()

api_calls = 0

for ticker in tqdm(tickers):

    last_date = last_dates_dict.get(ticker)

    if last_date:
        start_dt = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    else:
        start_dt = pd.Timestamp("2018-01-01")

    # 今日以降なら取得しない
    if start_dt >= today:
        continue

    start_date = start_dt.strftime("%Y-%m-%d")

    try:

        api_calls += 1

        data = yf.download(
            ticker,
            start=start_date,
            progress=False
        )

        if data.empty:
            continue

        # マルチインデックス解消
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data["Ticker"] = ticker

        data = data[
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Ticker",
            ]
        ]

        if last_date:
            data = data[data["Date"] > pd.to_datetime(last_date)]

        if not data.empty:
            dfs.append(data)

    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        continue

# =========================
# Parquetに追加保存
# =========================
if dfs:

    df_new = pd.concat(dfs, ignore_index=True)

    df_all = pd.concat([df_existing, df_new], ignore_index=True)

    df_all = df_all.drop_duplicates(subset=["Date", "Ticker"])

    df_all.to_parquet(PARQUET_FILE, index=False)

    print("追加行数:", len(df_new))

else:

    print("追加データなし")

# =========================
# 保存確認
# =========================
df_check = con.execute(
    f"""
    SELECT COUNT(*) FROM '{PARQUET_FILE}'
    """
).fetchone()[0]

print("保存完了 行数:", df_check)

print("API呼び出し回数:", api_calls)

con.close()
