import streamlit as st
import pandas as pd
import os

st.title("毎日更新♪株予測AI(*^^*)")

csv_path = "today_picks.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.subheader("5日後に上がりそうな5銘柄")
    st.dataframe(df)
else:
    st.warning("CSV がまだ生成されていません。GitHub Actions で Run Daily を先に実行してください。")
