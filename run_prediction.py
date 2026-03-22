import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRegressor
from datetime import datetime
from pandas.tseries.offsets import BDay


# =========================
# 設定
# =========================
TOP_N = 5
WEAK_TOP_PERCENT = 0.01  # ⭐追加（上位1%）

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PERF_LOG = os.path.join(BASE_DIR, "logs/performance.csv")


FEATURES = [
    "Return_1_rank",
    "MA5_ratio_rank",
    "MA25_ratio_rank",
    "MA75_ratio_rank",
    "Volatility_rank",
    "Volume_change_rank",
    "HL_range_rank",
    "RSI_rank"
]

TARGET = "Target"


# =========================
# 戦略
# =========================
def generate_global_strategy():
    return """
========================
■ AI戦略の前提（重要）
========================

■ エントリー
当日の寄付き or 引けで即エントリー
■ 保有期間
5営業日
■ 利確
5営業日後に決済
■ 損切り
-3%

------------------------
■ 注意事項
本戦略は短期トレードです
"""


# =========================
# レジーム判定
# =========================
def get_regime(score):
    if score > 0.55:
        return "strong"
    elif score > 0.52:
        return "slightly_strong"
    elif score > 0.5:
        return "neutral"
    else:
        return "weak"


# =========================
# 最適銘柄数
# =========================
def get_best_n(regime):

    if not os.path.exists(PERF_LOG):
        return 3, "（実績なし）"

    df = pd.read_csv(PERF_LOG)
    df_r = df[df["regime"] == regime]

    if df_r.empty or "n" not in df_r.columns:
        return 3, "（データ不足）"

    best_n = 3
    best_sharpe = -999

    for n in sorted(df_r["n"].unique()):
        df_n = df_r[df_r["n"] == n]

        if len(df_n) < 10:
            continue

        avg = df_n["avg_return"].mean()
        std = df_n["avg_return"].std()
        sharpe = avg / std if std != 0 else 0

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_n = n

    return best_n, f"最適銘柄数: {best_n}（Sharpe最大）"


# =========================
# 実績
# =========================
def get_performance_comment(regime):

    if not os.path.exists(PERF_LOG):
        return "\n（実績データなし）\n"

    df = pd.read_csv(PERF_LOG)
    df_r = df[df["regime"] == regime]

    if len(df_r) < 10:
        return "\n（データ不足）\n"

    win = df_r["win"].mean()
    avg = df_r["avg_return"].mean()

    return f"""
■ 過去類似局面の実績
勝率: {win:.2%}
平均リターン: {avg:.2%}
"""


# =========================
# 日次判断（★ここが変更ポイント）
# =========================
def generate_daily_decision(picks_df, full_df):

    market_score = full_df["Pred"].mean()
    regime = get_regime(market_score)

    best_n, best_comment = get_best_n(regime)

    # =========================
    # ⭐弱気市場の特別ルール追加
    # =========================
    weak_exception_mode = False
    weak_picks = pd.DataFrame()

    if regime == "weak":

        weak_exception_mode = True

        # 上位1%
        threshold = full_df["Pred"].quantile(1 - WEAK_TOP_PERCENT)
        top = full_df[full_df["Pred"] >= threshold].copy()

        # 出来高急増（rank上位）
        top = top[top["Volume_change_rank"] <= 10]

        weak_picks = top.sort_values("Pred", ascending=False)

    # =========================
    # 表示
    # =========================
    if regime == "strong":
        trend = "強気"
        action = "積極エントリー"
        pos = "80〜100%"
    elif regime == "slightly_strong":
        trend = "やや強気"
        action = "選別エントリー"
        pos = "50〜70%"
    elif regime == "neutral":
        trend = "中立"
        action = "様子見"
        pos = "30%"
    else:
        trend = "弱気"
        action = "原則見送り（例外あり）"
        pos = "0%〜限定"

    perf_text = get_performance_comment(regime)

    # =========================
    # 弱気注釈追加
    # =========================
    weak_text = ""
    if weak_exception_mode:
        weak_text = f"""

------------------------
■ 弱気市場の例外ルール
・Pred上位1%
・出来高急増銘柄のみ
→ リスク高・限定エントリー

■ 該当銘柄数: {len(weak_picks)}
"""

    text = f"""

========================
■ 本日のAI判断
========================

市場評価: {trend}
行動: {action}
推奨ポジション: {pos}

------------------------

■ 今日の最適銘柄数
{best_n}銘柄（過去実績ベース）

------------------------

{perf_text}

{weak_text}
"""

    return text, regime, best_n, weak_picks


# =========================
# 記事生成（弱気対応）
# =========================
def generate_article(df, daily_comment, best_n, weak_picks, regime):

    texts = [daily_comment, generate_global_strategy()]

    # =========================
    # 弱気モード分岐
    # =========================
    if regime == "weak" and len(weak_picks) > 0:
        selected = weak_picks.head(best_n)
    else:
        selected = df.head(best_n)

    for _, row in selected.iterrows():
        texts.append(f"""
========================
{row["PredRank"]}位: {row["銘柄名"]} ({row["コード"]})
========================

■ AI評価
{row["PredRank"]}位
""")

    return "\n".join(texts)


# =========================
# データ読み込み
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
latest_date = df["Date"].max()


# =========================
# モデル
# =========================
retrain = (latest_date.weekday() == 0) or (not os.path.exists(MODEL_PATH))

if retrain:
    train = df[df["Date"] < latest_date]
    model = LGBMRegressor()
    model.fit(train[FEATURES], train[TARGET])

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
else:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


# =========================
# 今日データ
# =========================
today = df[df["Date"] == latest_date].copy()

today["Pred"] = model.predict(today[FEATURES])
today["Rank"] = today["Pred"].rank(ascending=False)


# =========================
# 銘柄名
# =========================
df_info = pd.read_csv(os.path.join(BASE_DIR, "data_j.csv"), dtype=str)
ticker_to_name = dict(zip(df_info["コード"], df_info["銘柄名"]))


# =========================
# picks
# =========================
picks = today.nsmallest(TOP_N, "Rank").copy()

picks["コード"] = picks["Ticker"].str.replace(".T", "", regex=False)
picks["銘柄名"] = picks["コード"].map(ticker_to_name)
picks["PredRank"] = range(1, len(picks) + 1)


# =========================
# 日次判断
# =========================
daily_comment, regime, best_n, weak_picks = generate_daily_decision(picks, today)


# =========================
# CSV
# =========================
picks[["Ticker", "コード", "銘柄名"]].to_csv("today_picks.csv", index=False)


# =========================
# 記事
# =========================
article = generate_article(picks, daily_comment, best_n, weak_picks, regime)

with open("note_article.txt", "w", encoding="utf-8") as f:
    f.write(article)


# =========================
# ログ
# =========================
today_dt = datetime.now()
target_dt = today_dt + BDay(5)

picks["predict_date"] = today_dt.strftime("%Y-%m-%d")
picks["target_date"] = target_dt.strftime("%Y-%m-%d")
picks["regime"] = regime

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

save_cols = [
    "Ticker",
    "コード",
    "銘柄名",
    "Pred",
    "PredRank",
    "predict_date",
    "target_date",
    "regime"
]

picks[save_cols].to_csv(os.path.join(LOG_DIR, "predictions.csv"), index=False)

print("✅ 完全処理完了")