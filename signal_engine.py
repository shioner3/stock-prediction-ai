import pandas as pd
import numpy as np

# =========================
# パラメータ
# =========================
TOP_N = 3

# =========================
# シグナル生成
# =========================
def generate_signals(df):

    df = df.copy()

    # =========================
    # 条件スコア
    # =========================
    score = np.zeros(len(df))

    # 初動モメンタム
    score += (df["return_3d"] > 0.05) * 1.0

    # 出来高
    score += (df["volume_ratio"] > 1.5) * 1.0

    # 過熱回避
    score += (df["ma5_diff"] < 0.1) * 0.5

    # ボラ
    score += (df["volatility_5"] > 0.01) * 0.5

    # クロスセクション
    score += (df["return_rank"] > 0.9) * 1.0

    score += (df["volume_rank"] > 0.8) * 0.5

    df["signal_score"] = score

    # =========================
    # 市場フィルタ
    # =========================
    df = df[df["market_trend_5"] > 0]

    # =========================
    # 日次ランキング
    # =========================
    df["rank"] = (
        df.groupby("Date")["signal_score"]
        .rank(ascending=False, method="first")
    )

    # =========================
    # 上位抽出
    # =========================
    df["signal"] = df["rank"] <= TOP_N

    return df