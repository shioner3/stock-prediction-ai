import subprocess
import os
import duckdb

def run_script(script_name):
    print(f"Running {script_name}...")
    script_path = os.path.join(os.getcwd(), script_name)
    subprocess.run(["python", script_path], check=True)

DB_FILE = "stock.db"

# =========================
# DB初期化
# =========================

if not os.path.exists(DB_FILE):

    print("DuckDB not found. Creating database...")

    con = duckdb.connect(DB_FILE)

    con.execute("""
    CREATE TABLE IF NOT EXISTS prices(
        Date DATE,
        Ticker VARCHAR,
        Open DOUBLE,
        High DOUBLE,
        Low DOUBLE,
        Close DOUBLE,
        Volume DOUBLE
    )
    """)

    con.close()

# =========================
# ① 株価データ取得（差分更新）
# =========================

run_script("download_prices.py")

# =========================
# ② 特徴量作成
# =========================

run_script("feature_engineering.py")

# =========================
# ③ 予測
# =========================

run_script("run_prediction.py")

print("Pipeline finished.")
