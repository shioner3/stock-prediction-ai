import subprocess
import os

PARQUET_FILE = "stock_data/prices.parquet"
ARTICLE_FILE = "note_article.txt"


def run_script(script_name):
    print(f"Running {script_name}...")
    script_path = os.path.join(os.getcwd(), script_name)
    subprocess.run(["python", script_path], check=True)


# =========================
# 1️⃣ Parquet確認
# =========================
if not os.path.exists(PARQUET_FILE):
    print("Parquet not found. It will be created by download_prices.py")

# =========================
# 2️⃣ 株価データ取得
# =========================
run_script("download_prices.py")

# =========================
# 3️⃣ 特徴量作成
# =========================
run_script("feature_engineering.py")

# =========================
# 4️⃣ 予測
# =========================
run_script("run_prediction.py")

# =========================
# 5️⃣ 🆕 記事生成
# =========================
def generate_article():
    print("📝 記事生成開始...")

    # run_prediction.py が出した結果を使う想定
    result_path = "today_picks.csv"

    if not os.path.exists(result_path):
        raise Exception("today_picks.csv が存在しません")

    # CSV読み込み
    import pandas as pd
    df = pd.read_csv(result_path)

    article = []
    article.append("【AI株式分析レポート】\n")
    article.append("本日の注目銘柄TOPリスト\n")

    for i, row in df.iterrows():
        ticker = row["ticker"]
        score = row["score"]
        ret = row["pred_return"]

        article.append(f"""
■ {i+1}. {ticker}
AIスコア: {score:.3f}
期待リターン: {ret*100:.2f}%

→ 短期的に注目度の高い銘柄です
""")

    text = "\n".join(article)

    with open(ARTICLE_FILE, "w", encoding="utf-8") as f:
        f.write(text)

    print("📝 note_article.txt 生成完了")


generate_article()

# =========================
# 6️⃣ サイズチェック
# =========================
size = os.path.getsize(PARQUET_FILE)
print("Parquet size:", size)

if size < 1000000:
    raise Exception("Parquet破損の可能性あり。保存停止。")

print("Pipeline finished.")
