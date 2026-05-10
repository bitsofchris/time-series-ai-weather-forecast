"""Plotly figure builders for the Toto weather demo."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .forecast import TotoForecast


def metric_figure(
    history: pd.Series,
    toto: TotoForecast,
    nws: pd.Series | None,
    title: str,
    y_label: str,
    now: pd.Timestamp | None = None,
) -> go.Figure:
    fig = go.Figure()

    # Past actuals
    fig.add_trace(
        go.Scatter(
            x=history.index, y=history.values,
            name="Ecowitt (past)", mode="lines",
            line=dict(color="#222", width=2),
        )
    )

    # Toto p10–p90 band
    fig.add_trace(
        go.Scatter(
            x=list(toto.p90.index) + list(toto.p10.index[::-1]),
            y=list(toto.p90.values) + list(toto.p10.values[::-1]),
            fill="toself", fillcolor="rgba(31,119,180,0.18)",
            line=dict(width=0), hoverinfo="skip",
            name="Toto 10–90% interval",
        )
    )
    # Toto median
    fig.add_trace(
        go.Scatter(
            x=toto.median.index, y=toto.median.values,
            name="Toto median", mode="lines",
            line=dict(color="#1f77b4", width=2, dash="dash"),
        )
    )

    if nws is not None and not nws.empty:
        fig.add_trace(
            go.Scatter(
                x=nws.index, y=nws.values,
                name="NWS forecast", mode="lines",
                line=dict(color="#d62728", width=2, dash="dot"),
            )
        )

    if now is not None:
        fig.add_vline(x=now, line=dict(color="#888", dash="dot", width=1))

    fig.update_layout(
        title=title,
        xaxis_title="Time (UTC)",
        yaxis_title=y_label,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
