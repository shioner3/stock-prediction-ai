import pandas as pd
import yfinance as yf
from tqdm import tqdm
import os

CSV_FILE = "data_j.csv"
PARQUET_FILE = "stock_data/prices.parquet"

# =========================
# 🔥 正規化関数（最重要）
# =========================
def normalize_ticker(t):
    return str(t).strip().upper()

# =========================
# 銘柄マスタ読み込み
# =========================
df_list = pd.read_csv(CSV_FILE, dtype=str)

df_list = df_list[
    df_list["市場・商品区分"].str.contains(
        "プライム|スタンダード|グロース", na=False
    )
]

# 🔥 完全正規化
df_list["Ticker"] = df_list["コード"].apply(normalize_ticker) + ".T"
df_list["Name"] = df_list["銘柄名"].astype(str).str.strip()

tickers = df_list["Ticker"].tolist()

# Name辞書
name_dict = dict(zip(df_list["Ticker"], df_list["Name"]))

print("銘柄数:", len(tickers))

# =========================
# データフォルダ
# =========================
os.makedirs("stock_data", exist_ok=True)

# =========================
# 既存データ読み込み
# =========================
if os.path.exists(PARQUET_FILE):

    df_existing = pd.read_parquet(PARQUET_FILE)

    # 🔥 正規化（ここが最重要）
    df_existing["Ticker"] = df_existing["Ticker"].apply(normalize_ticker)
    df_existing["Date"] = pd.to_datetime(df_existing["Date"]).dt.tz_localize(None)

    # Name補完（過去データにも適用）
    df_existing["Name"] = df_existing["Ticker"].map(name_dict).fillna(df_existing.get("Name"))

else:
    df_existing = pd.DataFrame(
        columns=["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Name"]
    )

# =========================
# 最新日取得
# =========================
if not df_existing.empty:
    last_dates_df = (
        df_existing.groupby("Ticker")["Date"]
        .max()
        .reset_index()
        .rename(columns={"Date": "last_date"})
    )
else:
    last_dates_df = pd.DataFrame(columns=["Ticker", "last_date"])

last_dates_dict = dict(zip(last_dates_df["Ticker"], last_dates_df["last_date"]))

# =========================
# デバッグ（超重要）
# =========================
missing = sum([1 for t in tickers if t not in last_dates_dict])

print("=== DEBUG ===")
print("Existing rows:", len(df_existing))
print("missing ticker count:", missing)
print("sample existing:", list(last_dates_dict.keys())[:5])
print("sample new:", tickers[:5])
print("==============")

# =========================
# 差分取得
# =========================
dfs = []

today = pd.Timestamp.utcnow() + pd.Timedelta(hours=9)
today = today.normalize().tz_localize(None)

api_calls = 0

for ticker in tqdm(tickers):

    last_date = last_dates_dict.get(ticker)

    # =========================
    # 🔥 差分判定（安全版）
    # =========================
    if last_date is not None:
        last_date = pd.to_datetime(last_date, errors="coerce")

        if pd.notna(last_date):
            last_date = last_date.tz_localize(None)

            # 🔥 1日以内ならスキップ（安全）
            if last_date >= today - pd.Timedelta(days=1):
                continue

            start_dt = last_date + pd.Timedelta(days=1)
        else:
            start_dt = pd.Timestamp("2018-01-01")

    else:
        start_dt = pd.Timestamp("2018-01-01")

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

        # MultiIndex対策
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        # 🔥 Ticker / Name付与
        data["Ticker"] = ticker
        data["Name"] = name_dict.get(ticker, None)

        data = data[
            ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Name"]
        ]

        # 🔥 念のため差分フィルタ
        if last_date is not None:
            data = data[data["Date"] > last_date]

        if not data.empty:
            dfs.append(data)

    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        continue

# =========================
# 保存
# =========================
if dfs:

    df_new = pd.concat(dfs, ignore_index=True)

    df_all = pd.concat([df_existing, df_new], ignore_index=True)

    # 🔥 重複削除（安全）
    df_all = df_all.drop_duplicates(subset=["Date", "Ticker"], keep="last")

    df_all.to_parquet(PARQUET_FILE, index=False)

    print("追加行数:", len(df_new))

else:
    print("追加データなし")

# =========================
# 確認
# =========================
final_rows = len(df_existing) + (len(df_new) if dfs else 0)

print("保存完了 行数:", final_rows)
print("API呼び出し回数:", api_calls)