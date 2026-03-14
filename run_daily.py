import subprocess
import os
import pandas as pd

def run_script(script_name):
    print(f"Running {script_name}...")
    script_path = os.path.join(os.getcwd(), script_name)
    subprocess.run(["python3", script_path], check=True)

# 1️⃣ データ取得
run_script("download_prices.py")

# 2️⃣ 初回だけ Parquet を作る
parquet_path = "data/japan_stock.parquet"
if not os.path.exists(parquet_path):
    print("Parquet file not found. Creating from CSV...")
    df_csv = pd.read_csv("data/data_j.csv", dtype=str)
    # 仮に日付カラムや必要カラムを入れる
    if 'Date' not in df_csv.columns:
        from datetime import datetime
        df_csv['Date'] = datetime.today().strftime("%Y-%m-%d")
    df_csv.to_parquet(parquet_path)
    print("Parquet created:", parquet_path)

# 3️⃣ 特徴量作成
run_script("feature_engineering.py")

# 4️⃣ 予測
run_script("run_prediction.py")

print("Done.")
