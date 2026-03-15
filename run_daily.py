import subprocess
import os
import duckdb

DB_FILE = "stock.db"

def run_script(script_name):
    print(f"Running {script_name}...")
    script_path = os.path.join(os.getcwd(), script_name)
    subprocess.run(["python", script_path], check=True)

# =========================
# 1️⃣ DuckDB初期化（初回のみ）
# =========================
if not os.path.exists(DB_FILE):
    print("DuckDB not found. Creating empty database...")
    con = duckdb.connect(DB_FILE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            Date DATE,
            Ticker VARCHAR,
            Open DOUBLE,
            High DOUBLE,
            Low DOUBLE,
            Close DOUBLE,
            Volume BIGINT
        )
    """)
    con.close()
    print("Empty DB created.")

# =========================
# 2️⃣ 株価データ取得（差分更新）
# =========================
run_script("download_prices.py")

# =========================
# 3️⃣ 特徴量作成
run_script("feature_engineering.py")

# =========================
# 4️⃣ 予測
run_script("run_prediction.py")

print("Pipeline finished.")

# =========================
# 5️⃣ DBサイズチェック
# =========================
size = os.path.getsize(DB_FILE)
print("DB size:", size)

if size < 1000000:
    raise Exception("DB破損の可能性あり。保存停止。")
