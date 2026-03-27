import pandas as pd
import yfinance as yf
from tqdm import tqdm
import os
import time

CSV_FILE = "data_j.csv"
PARQUET_FILE = "stock_data/prices.parquet"

RECENT_DAYS = 10
MIN_COUNT = 3000

BATCH_SIZE = 50        # ←小さくするほど安定
SLEEP_TIME = 1         # ←API保護
RETRY = 3              # ←再試行回数

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
# 🔥 バッチ取得（最重要）
# =========================
def fetch_batch(batch):

    for attempt in range(RETRY):
        try:
            data = yf.download(
                batch,
                period=f"{RECENT_DAYS}d",
                group_by="ticker",
                auto_adjust=True,
                threads=False,   # ←絶対オフ
                progress=False
            )
            return data
        except Exception as e:
            print(f"Retry {attempt+1}:", e)
            time.sleep(2)

    return None

# =========================
# 🔥 直近データ取得
# =========================
print("\n=== 直近データ取得（安定版） ===")

dfs_recent = []

for i in tqdm(range(0, len(tickers), BATCH_SIZE)):

    batch = tickers[i:i+BATCH_SIZE]
    data = fetch_batch(batch)

    if data is None:
        continue

    for ticker in batch:
        try:
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

            dfs_recent.append(df_t)

        except Exception:
            continue

    time.sleep(SLEEP_TIME)

# =========================
# 🔥 結合
# =========================
if len(dfs_recent) == 0:
    print("⚠ データ取得失敗")
    exit()

df_recent = pd.concat(dfs_recent, ignore_index=True)

df_all = pd.concat([df_existing, df_recent], ignore_index=True)

df_all = df_all.drop_duplicates(
    subset=["Date", "Ticker"],
    keep="last"
)

# =========================
# 🔥 完全性チェック
# =========================
counts = df_all["Date"].value_counts()

print("\n=== 最新付近チェック ===")
print(counts.sort_index().tail(10))

valid_dates = counts[counts >= MIN_COUNT].index

if len(valid_dates) == 0:
    print("⚠ 有効な日なし")
    latest_valid = None
else:
    latest_valid = max(valid_dates)

print("\n=== 判定 ===")
print("最新日:", df_all["Date"].max())
print("有効最新日:", latest_valid)
print("銘柄数:", counts.get(latest_valid, 0))

# =========================
# 保存
# =========================
df_all.to_parquet(PARQUET_FILE, index=False)

# =========================
# 結果
# =========================
print("\n=== 完了 ===")
print("総行数:", len(df_all))