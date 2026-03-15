import subprocess
import os
import pandas as pd

def run_script(script_name):
    print(f"Running {script_name}...")
    script_path = os.path.join(os.getcwd(), script_name)
    subprocess.run(["python", script_path], check=True)

parquet_path = "japan_stock.parquet"

# ① Parquetが無ければ初期作成
if not os.path.exists(parquet_path):
    print("Parquet file not found. Creating empty base...")
    df = pd.DataFrame()
    df.to_parquet(parquet_path)

# ② 株価データ取得（差分更新）
run_script("download_prices.py")

# ③ 特徴量作成
run_script("feature_engineering.py")

# ④ 予測
run_script("run_prediction.py")

print("Done.")
