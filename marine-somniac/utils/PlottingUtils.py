import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure


class PlottingUtils:
    @staticmethod
    def plot_label_trace(df):
        trace = go.Bar(
            x=df['second'], y=[1 for _ in range(len(df))],
            marker_color=df['color'].to_list(), name='Labels'
        )
        return trace

    @staticmethod
    def plot_feature(df: pd.DataFrame, label_trace, plot='line') -> Figure:
        plot = {
            'line': px.line,
            'scatter': px.scatter
        }[plot]

        n_rows = 1 if df is None else len(df.columns)
        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True
        )
        if df is not None:
            for i, col in enumerate([c for c in df.columns if c != 'time']):
                fig.add_trace(
                    go.Scatter(x=df['time'], y=df[col], name=col),
                    row=i+1, col=1
                )
        fig.add_trace(label_trace, row=n_rows, col=1)
        fig.update_layout(margin={'t':0, 'b': 0}, height=300*n_rows)
        return fig
    