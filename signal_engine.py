import pandas as pd
import numpy as np

TOP_N = 2

def generate_signals(df):

    df = df.copy()

    # フィルタ（重要）
    df = df[
        (df["return_3d"] > 0.03) &
        (df["volume_ratio"] > 1.2) &
        (df["return_rank"] > 0.8) &
        (df["ma5_diff"] < 0.1)
    ]

    # スコア
    df["signal_score"] = (
        df["return_3d"] * 10 +
        np.log1p(df["volume_ratio"]) * 2 +
        df["return_rank"] * 2
    )

    # ランク
    df["rank"] = df.groupby("Date")["signal_score"]\
        .rank(ascending=False, method="first")

    df["signal"] = df["rank"] <= TOP_N

    return df