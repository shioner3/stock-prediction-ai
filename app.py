import streamlit as st
import pandas as pd

st.title("毎日更新♪株予測AI(*^^*)")

try:
    df = pd.read_csv("today_picks.csv")
    st.subheader("5日後に上がりそうな5銘柄")
    st.dataframe(df)
except FileNotFoundError:
    st.warning("CSV がまだ生成されていません。GitHub Actions で Run Daily を先に実行してください。")
