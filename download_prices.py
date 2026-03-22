import pandas as pd
import yfinance as yf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support
import time
import random

# =========================
# 設定
# =========================
CSV_FILE = "data_j.csv"  # 銘柄マスタ（コード + Name）
PARQUET_FILE = "stock_data/prices.parquet"

DAYS = 5
TIMEOUT = 10
RETRY_LIMIT = 5
WAIT_TIME = 10
MAX_SLEEP_TIME = 3

# =========================
# 銘柄コード正規化
# =========================
def normalize_ticker(ticker):
    if not ticker.endswith(".T"):
        ticker += ".T"
    return ticker

# =========================
# 株価取得
# =========================
def fetch_price(row):
    ticker = normalize_ticker(row["コード"])
    name = row["銘柄名"]

    retry_count = 0

    while retry_count < RETRY_LIMIT:
        try:
            data = yf.download(
                ticker,
                period=f"{DAYS}d",
                interval="1d",
                timeout=TIMEOUT,
                progress=False
            )

            if data.empty:
                return None

            data = data.reset_index()

            # =========================
            # メタ情報付与（重要）
            # =========================
            data["Ticker"] = ticker
            data["Name"] = name

            # 列整理
            data = data[[
                "Date",
                "Ticker",
                "Name",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]]

            return data

        except Exception as e:
            print(f"エラー {ticker}: {e}")
            retry_count += 1
            time.sleep(WAIT_TIME)

        time.sleep(random.uniform(1, MAX_SLEEP_TIME))

    return None

# =========================
# メイン
# =========================
def main():

    # 銘柄マスタ読み込み
    df_list = pd.read_csv(CSV_FILE, dtype=str)

    # 市場フィルタ
    df_list = df_list[
        df_list["市場・商品区分"].str.contains(
            "プライム|スタンダード|グロース",
            na=False
        )
    ]

    print(f"対象銘柄数: {len(df_list)}")

    # multiprocessing用にdict化
    rows = df_list[["コード", "銘柄名"]].to_dict("records")

    # =========================
    # 並列取得
    # =========================
    results = []

    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(fetch_price, rows), total=len(rows)):
            if res is not None:
                results.append(res)

    # =========================
    # 結合
    # =========================
    df = pd.concat(results, ignore_index=True)

    print("最終データサイズ:", df.shape)
    print(df.head())

    # =========================
    # 保存
    # =========================
    df.to_parquet(PARQUET_FILE)

    print("Parquet保存完了:", PARQUET_FILE)


if __name__ == "__main__":
    freeze_support()
    main()