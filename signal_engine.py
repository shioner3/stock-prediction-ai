import pandas as pd
import numpy as np

TOP_N = 5

def generate_signals(df):

    df = df.copy()

    # 超シンプルスコア
    df["signal_score"] = (
        df["return_3d"].clip(lower=0) * 10 +
        df["volume_ratio"].clip(lower=0)
    )

    # フィルタほぼなし（まず動かす）
    df = df[df["return_3d"] > 0]

    # ランク
    df["rank"] = df.groupby("Date")["signal_score"]\
        .rank(ascending=False, method="first")

    # 上位
    df["signal"] = df["rank"] <= TOP_N

    return df