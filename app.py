import gradio as gr
import pandas as pd
import os

def load_data():
    if os.path.exists("today_picks.csv"):
        df = pd.read_csv("today_picks.csv")
        return df
    else:
        return "CSVがまだありません"

demo = gr.Interface(
    fn=load_data,
    inputs=None,
    outputs="dataframe",
    title="毎日更新♪株予測AI(*^^*)",
    description="5日後に上がりそうな5銘柄"
)

if __name__ == "__main__":
    demo.launch()
