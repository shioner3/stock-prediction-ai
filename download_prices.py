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

# 🔥 ticker統一（安全版）
tickers = df_list["コード"].astype(str).tolist()
tickers = [t if t.endswith(".T") else t + ".T" for t in tickers]

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

    # 🔥 Ticker統一（最重要）
    df_existing["Ticker"] = df_existing["Ticker"].astype(str)
    df_existing["Ticker"] = df_existing["Ticker"].apply(
        lambda x: x if x.endswith(".T") else x + ".T"
    )

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

# =========================
# 🔍 デバッグ（超重要）
# =========================
print("=== DEBUG ===")
print("Existing rows:", len(df_existing))
print("Ticker sample:", df_existing["Ticker"].unique()[:5] if not df_existing.empty else "EMPTY")
print("Last dates sample:")
print(last_dates_df.head())
print("==============")

print("取得対象のtickers例:")
print(tickers[:5])

# =========================
# 差分取得
# =========================
dfs = []

# 🔥 UTCに統一（重要）
today = pd.Timestamp.utcnow().normalize()

api_calls = 0

for ticker in tqdm(tickers):

    last_date = last_dates_dict.get(ticker)

    if last_date:
        start_dt = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    else:
        start_dt = pd.Timestamp("2018-01-01")

    # 🔥 条件緩和
    if start_dt > today:
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

        # 🔥 念のため差分フィルタ
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