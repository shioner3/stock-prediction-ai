import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(
    page_title="株予測AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 毎日更新♪ 株予測AI (*^^*)")

st.autorefresh(interval=60000)

# =========================
# CSV確認
# =========================

if os.path.exists("today_picks.csv"):

    df = pd.read_csv("today_picks.csv")

    st.subheader("📊 5日後に上がりそうな5銘柄")

    # ---------------------
    # テーブル表示
    # ---------------------

    st.dataframe(df, use_container_width=True)

    # 更新時間
    update_time = datetime.fromtimestamp(
        os.path.getmtime("today_picks.csv")
    )

    st.caption(f"最終更新: {update_time}")

    st.divider()

    # ---------------------
    # 銘柄カード表示
    # ---------------------

    st.subheader("🔎 銘柄詳細")

    cols = st.columns(len(df))

    for i, row in df.iterrows():

        ticker = row["Ticker"]

        with cols[i]:

            st.markdown(f"### {ticker}")

            # 上昇スコア（Predがある場合）
            if "Pred" in df.columns:

                score = float(row["Pred"])

                st.progress(
                    min(max(score, 0), 1)
                )

                st.caption(f"AIスコア: {score:.2f}")

            # Yahooリンク
            url = f"https://finance.yahoo.co.jp/quote/{ticker}"

            st.markdown(
                f"[📊 Yahoo Finance]({url})"
            )

            # チャート
            chart_url = f"https://finance.yahoo.co.jp/quote/{ticker}/chart"

            st.markdown(
                f"[📈 チャートを見る]({chart_url})"
            )

else:

    st.warning(
        "CSVがまだ生成されていません。\n\n"
        "GitHub Actionsを実行してください。"
    )
