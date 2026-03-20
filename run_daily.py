import os
import shutil
from datetime import datetime

# =========================
# 設定
# =========================
OUTPUT_DIR = "daily_reports"

# =========================
# 実行
# =========================
def main():
    today = datetime.now().strftime("%Y-%m-%d")

    print("🚀 日次処理開始:", today)

    # =========================
    # ① 予測スクリプト実行
    # =========================
    print("📊 予測実行中...")
    os.system("python run_prediction.py")

    # =========================
    # ② 出力確認
    # =========================
    if not os.path.exists("today_picks.csv"):
        print("❌ today_picks.csv が見つかりません")
        return

    if not os.path.exists("note_article.txt"):
        print("❌ note_article.txt が見つかりません")
        return

    # =========================
    # ③ フォルダ作成
    # =========================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================
    # ④ 日付付き保存
    # =========================
    csv_name = f"{OUTPUT_DIR}/{today}_picks.csv"
    note_name = f"{OUTPUT_DIR}/{today}_article.txt"

    shutil.copy("today_picks.csv", csv_name)
    shutil.copy("note_article.txt", note_name)

    print("✅ 保存完了:")
    print(csv_name)
    print(note_name)

    # =========================
    # ⑤ ログ（簡易実績用）
    # =========================
    with open(f"{OUTPUT_DIR}/log.txt", "a", encoding="utf-8") as f:
        f.write(f"{today} 実行完了\n")

    print("🎉 日次処理完了")


if __name__ == "__main__":
    main()
