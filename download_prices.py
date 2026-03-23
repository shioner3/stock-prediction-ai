import pandas as pd
import yfinance as yf
from tqdm import tqdm
import os

CSV_FILE = "data_j.csv"
PARQUET_FILE = "stock_data/prices.parquet"

# =========================
# 🔥 正規化関数
# =========================
def normalize_ticker(t):
    return str(t).strip().upper()

# =========================
# 銘柄マスタ
# =========================
df_list = pd.read_csv(CSV_FILE, dtype=str)

df_list = df_list[
    df_list["市場・商品区分"].str.contains(
        "プライム|スタンダード|グロース", na=False
    )
]

df_list["Ticker"] = df_list["コード"].apply(normalize_ticker) + ".T"
df_list["Name"] = df_list["銘柄名"].astype(str).str.strip()

tickers = df_list["Ticker"].tolist()
name_dict = dict(zip(df_list["Ticker"], df_list["Name"]))

print("銘柄数:", len(tickers))

# =========================
# フォルダ
# =========================
os.makedirs("stock_data", exist_ok=True)

# =========================
# 既存データ
# =========================
if os.path.exists(PARQUET_FILE):

    df_existing = pd.read_parquet(PARQUET_FILE)

    df_existing["Ticker"] = df_existing["Ticker"].apply(normalize_ticker)
    df_existing["Date"] = pd.to_datetime(df_existing["Date"], errors="coerce").dt.tz_localize(None)

    # 🔥 Date壊れ対策（超重要）
    df_existing = df_existing.dropna(subset=["Date"])

    # 🔥 Name補完（安全）
    mapped_name = df_existing["Ticker"].map(name_dict)

    if "Name" in df_existing.columns:
        df_existing["Name"] = mapped_name.combine_first(df_existing["Name"])
    else:
        df_existing["Name"] = mapped_name

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
else:
    last_dates_df = pd.DataFrame(columns=["Ticker", "last_date"])

last_dates_dict = dict(zip(last_dates_df["Ticker"], last_dates_df["last_date"]))

# =========================
# デバッグ
# =========================
missing = sum([1 for t in tickers if t not in last_dates_dict])

print("=== DEBUG ===")
print("Existing rows:", len(df_existing))
print("missing ticker count:", missing)
print("==============")

# =========================
# 差分取得
# =========================
dfs = []

today = pd.Timestamp.utcnow() + pd.Timedelta(hours=9)
today = today.normalize().tz_localize(None)

api_calls = 0
skip_count = 0

for ticker in tqdm(tickers):

    last_date = last_dates_dict.get(ticker)

    # =========================
    # 🔥 差分判定（完全修正版）
    # =========================
    if last_date is not None:
        last_date = pd.to_datetime(last_date, errors="coerce")

        # 🔥 NaTならスキップ（フル取得防止）
        if pd.isna(last_date):
            skip_count += 1
            continue

        last_date = last_date.tz_localize(None)

        # 🔥 最新ならスキップ
        if last_date >= today - pd.Timedelta(days=1):
            skip_count += 1
            continue

        start_dt = last_date + pd.Timedelta(days=1)

    else:
        # 🔥 初回のみフル取得
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

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        data["Ticker"] = ticker
        data["Name"] = name_dict.get(ticker, None)

        data = data[
            ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Name"]
        ]

        # 念のため差分
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

    df_all = df_all.drop_duplicates(subset=["Date", "Ticker"], keep="last")

    df_all.to_parquet(PARQUET_FILE, index=False)

    print("追加行数:", len(df_new))

else:
    print("追加データなし")

# =========================
# 結果
# =========================
print("保存完了 行数:", len(df_existing) + (len(dfs) if dfs else 0))
print("API呼び出し回数:", api_calls)
print("スキップ数:", skip_count)