import pandas as pd
import yfinance as yf
from tqdm import tqdm
import os
import time

# =========================
# ファイル
# =========================
CSV_FILE = "data_j.csv"
OUTPUT_FILE = "stock_data/shares_outstanding.parquet"

SLEEP_TIME = 0.2

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
# 銘柄一覧
# =========================
df_list = pd.read_csv(
    CSV_FILE,
    dtype=str
)

df_list = df_list[
    df_list["市場・商品区分"]
    .str.contains(
        "プライム|スタンダード|グロース",
        na=False
    )
]

df_list["Ticker"] = (
    df_list["コード"]
    .apply(normalize_ticker)
)

df_list["Name"] = (
    df_list["銘柄名"]
    .astype(str)
    .str.strip()
)

df_list["Industry"] = (
    df_list["33業種区分"]
    .astype(str)
    .str.strip()
)

tickers = df_list["Ticker"].tolist()

name_dict = dict(
    zip(df_list["Ticker"], df_list["Name"])
)

industry_dict = dict(
    zip(df_list["Ticker"], df_list["Industry"])
)

print("対象銘柄数:", len(tickers))

# =========================
# 取得
# =========================
rows = []

for ticker in tqdm(tickers):

    shares = None

    try:

        tk = yf.Ticker(ticker)

        # 軽い方を優先
        try:
            shares = tk.fast_info.get("shares")
        except:
            pass

        # fallback
        if shares is None:

            try:
                shares = (
                    tk.info
                    .get("sharesOutstanding")
                )
            except:
                pass

        rows.append(
            {
                "Ticker": ticker,
                "Name": name_dict.get(ticker),
                "Industry": industry_dict.get(ticker),
                "SharesOutstanding": shares
            }
        )

    except Exception:

        rows.append(
            {
                "Ticker": ticker,
                "Name": name_dict.get(ticker),
                "Industry": industry_dict.get(ticker),
                "SharesOutstanding": None
            }
        )

    time.sleep(SLEEP_TIME)

# =========================
# DataFrame化
# =========================
df_shares = pd.DataFrame(rows)

df_shares["UpdateDate"] = (
    pd.Timestamp.today()
    .normalize()
)

# =========================
# 保存
# =========================
df_shares.to_parquet(
    OUTPUT_FILE,
    index=False
)

# =========================
# 確認
# =========================
success = (
    df_shares["SharesOutstanding"]
    .notna()
    .sum()
)

print("\n=== 完了 ===")

print(
    "取得成功:",
    success
)

print(
    "取得失敗:",
    len(df_shares) - success
)

print(
    "\n保存先:",
    OUTPUT_FILE
)

print("\nサンプル:")

print(
    df_shares.head()
)