# apps/streamlit/plots_view.py

from typing import Dict
import numpy as np
import plotly.graph_objects as go


def make_time_series(cfg: Dict, T: np.ndarray, X: np.ndarray) -> go.Figure:
    fig = go.Figure()
    if cfg["sys"] == "single":
        fig.add_trace(go.Scatter(x=T, y=X[:, 0], mode="lines", name="θ(t)"))
        fig.add_trace(go.Scatter(x=T, y=X[:, 1], mode="lines", name="ω(t)"))
    else:
        fig.add_trace(go.Scatter(x=T, y=X[:, 0], mode="lines", name="θ₁(t)"))
        fig.add_trace(go.Scatter(x=T, y=X[:, 2], mode="lines", name="θ₂(t)"))
        fig.add_trace(go.Scatter(x=T, y=X[:, 1], mode="lines", name="ω₁(t)"))
        fig.add_trace(go.Scatter(x=T, y=X[:, 3], mode="lines", name="ω₂(t)"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def make_phase_space(cfg: Dict, X: np.ndarray) -> go.Figure:
    fig = go.Figure()
    if cfg["sys"] == "single":
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="lines", name="(θ, ω)"))
        fig.update_layout(xaxis_title="θ", yaxis_title="ω")
    else:
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="lines", name="(θ₁, ω₁)"))
        fig.add_trace(go.Scatter(x=X[:, 2], y=X[:, 3], mode="lines", name="(θ₂, ω₂)"))
        fig.update_layout(xaxis_title="θ, ω pairs", yaxis_title="")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def make_energy(T: np.ndarray, KE: np.ndarray, PE: np.ndarray, E: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=KE, mode="lines", name="T kinetic"))
    fig.add_trace(go.Scatter(x=T, y=PE, mode="lines", name="V potential"))
    fig.add_trace(go.Scatter(x=T, y=E, mode="lines", name="E total"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    return fig
