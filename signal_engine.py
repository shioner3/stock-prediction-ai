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
    # スコア（連続値）
    # =========================
    score = np.zeros(len(df))

    # 初動モメンタム（強いほど加点）
    score += np.clip(df["return_3d"], 0, 0.1) * 10

    # 出来高（対数で安定化）
    score += np.log1p(df["volume_ratio"]) * 2

    # 過熱回避（乖離が小さいほど良い）
    score += np.clip(0.1 - df["ma5_diff"], 0, 0.1) * 5

    # ボラ（低すぎも高すぎもNG → 中央寄せ）
    score += np.clip(df["volatility_5"], 0, 0.03) * 10

    # クロスセクション（かなり重要）
    score += df["return_rank"] * 2
    score += df["volume_rank"] * 1.5

    df["signal_score"] = score

    # =========================
    # 市場フィルタ（強化）
    # =========================
    df = df[df["market_trend_5"] > 0.001]

    # =========================
    # スコア最低ライン（重要）
    # =========================
    df = df[df["signal_score"] > 1.5]

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

    # =========================
    # 重み（後で使える）
    # =========================
    df["weight"] = (
        df["signal_score"] /
        df.groupby("Date")["signal_score"].transform("sum")
    )

    return df