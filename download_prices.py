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
        "プライム|スタンダード|グロース", na=False
    )
]

tickers = df_list["コード"].tolist()
tickers = [t + ".T" for t in tickers]  # DBでも同じ形式で保存

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
# tickerごとの最新日を取得
# =========================
last_dates_df = con.execute("""
SELECT Ticker, MAX(Date) AS last_date
FROM stock_prices
GROUP BY Ticker
""").fetchdf()
last_dates_dict = dict(zip(last_dates_df['Ticker'], last_dates_df['last_date']))

# 確認用出力
print("DBに保存されている各Tickerの最新日:")
print(last_dates_df.head())
print("取得対象のtickers例:")
print(tickers[:5])

# =========================
# 差分取得
# =========================
dfs = []

for ticker in tqdm(tickers):
    # DBにない場合は初期日付から
    last_date = last_dates_dict.get(ticker)
    start_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if last_date else "2018-01-01"

    try:
        data = yf.download(ticker, start=start_date, progress=False)
        if data.empty:
            # データなし（上場廃止や週末等）はスキップ
            continue

        # マルチインデックス解消
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data["Ticker"] = ticker
        data = data[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]]

        # DBにすでにある日付は削除
        if last_date:
            data = data[data["Date"] > pd.to_datetime(last_date)]

        if not data.empty:
            dfs.append(data)

    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        continue

# =========================
# DBに追加
# =========================
if dfs:
    df_new = pd.concat(dfs, ignore_index=True)
    df_new = df_new.drop_duplicates(subset=["Date", "Ticker"])

    con.register("temp_df", df_new)

    # すでにDBにある日付は挿入しない
    con.execute("""
    INSERT INTO stock_prices
    SELECT * FROM temp_df t
    WHERE NOT EXISTS (
        SELECT 1 FROM stock_prices s
        WHERE s.Ticker = t.Ticker AND s.Date = t.Date
    )
    """)

# =========================
# 保存確認
# =========================
count = con.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0]
print("保存完了:", count)

con.close()
