import pandas as pd
import yfinance as yf
from tqdm import tqdm
import duckdb
import os

CSV_FILE = "data_j.csv"
DB_FILE = "stock.db"

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
# DB接続
# =========================
con = duckdb.connect(DB_FILE)

# テーブル作成（初回のみ）
con.execute("""
CREATE TABLE IF NOT EXISTS stock_prices (
    Date DATE,
    Open DOUBLE,
    High DOUBLE,
    Low DOUBLE,
    Close DOUBLE,
    Volume BIGINT,
    Ticker TEXT
)
""")

# =========================
# 既存データ確認
# =========================
existing_data = con.execute("SELECT Ticker, Date FROM stock_prices").df()
existing_index = set(zip(existing_data["Ticker"], existing_data["Date"]))

if existing_data.empty:
    start_date = "2018-01-01"
else:
    last_date = con.execute("SELECT MAX(Date) FROM stock_prices").fetchone()[0]
    start_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

print("取得開始日:", start_date)

# =========================
# 差分取得
# =========================
dfs = []

for ticker in tqdm(tickers):
    try:
        data = yf.download(
            ticker,
            start=start_date,
            progress=False
        )

        if data.empty:
            continue

        # インデックスの階層を解消
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data["Ticker"] = ticker

        data = data[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]]

        # 既存DBと重複する行を除外
        data = data[~data.set_index(['Ticker', 'Date']).index.isin(existing_index)]

        if not data.empty:
            dfs.append(data.reset_index(drop=True))

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        continue

# =========================
# DBに追加
# =========================
if dfs:
    df_new = pd.concat(dfs, ignore_index=True)
    con.register("temp_df", df_new)

    con.execute("""
    INSERT INTO stock_prices
    SELECT *
    FROM temp_df
    """)

    # 重複削除
    con.execute("""
    CREATE OR REPLACE TABLE stock_prices AS
    SELECT DISTINCT *
    FROM stock_prices
    """)

# =========================
# 保存確認
# =========================
count = con.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0]
print("保存完了:", count)

con.close()
