"""
Plot helpers used by Streamlit systems.

- Consistent sizing/margins across pages
- Optional *cursor* (index) to highlight a current time sample
"""

from typing import Dict, Optional
import numpy as np
import plotly.graph_objects as go


def _clamp_idx(cursor_idx: Optional[int], n: int) -> Optional[int]:
    if cursor_idx is None or n <= 0:
        return None
    return int(max(0, min(n - 1, int(cursor_idx))))


def make_time_series(cfg: Dict, T: np.ndarray, X: np.ndarray, cursor_idx: Optional[int] = None) -> go.Figure:
    fig = go.Figure()
    cursor_idx = _clamp_idx(cursor_idx, len(T))

    if cfg.get("sys") == "single":
        fig.add_trace(go.Scatter(x=T, y=X[:, 0], mode="lines", name="θ(t)"))
        fig.add_trace(go.Scatter(x=T, y=X[:, 1], mode="lines", name="ω(t)"))

        if cursor_idx is not None:
            t = float(T[cursor_idx])
            fig.add_vline(x=t, line_width=1, line_dash="dash")
            fig.add_trace(go.Scatter(x=[t], y=[float(X[cursor_idx, 0])], mode="markers", showlegend=False))
            fig.add_trace(go.Scatter(x=[t], y=[float(X[cursor_idx, 1])], mode="markers", showlegend=False))
    else:
        fig.add_trace(go.Scatter(x=T, y=X[:, 0], mode="lines", name="θ₁(t)"))
        fig.add_trace(go.Scatter(x=T, y=X[:, 2], mode="lines", name="θ₂(t)"))
        fig.add_trace(go.Scatter(x=T, y=X[:, 1], mode="lines", name="ω₁(t)"))
        fig.add_trace(go.Scatter(x=T, y=X[:, 3], mode="lines", name="ω₂(t)"))

        if cursor_idx is not None:
            fig.add_vline(x=float(T[cursor_idx]), line_width=1, line_dash="dash")

    fig.update_layout(xaxis_title="t [s]", yaxis_title="Energy [J]", height=280, margin=dict(l=10, r=10, t=25, b=10))
    return fig


def make_phase_space(cfg: Dict, X: np.ndarray, cursor_idx: Optional[int] = None) -> go.Figure:
    fig = go.Figure()
    cursor_idx = _clamp_idx(cursor_idx, len(X))

    if cfg.get("sys") == "single":
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="lines", name="(θ, ω)"))
        fig.update_layout(xaxis_title="θ [rad]", yaxis_title="ω [rad/s]")

        if cursor_idx is not None:
            fig.add_trace(
                go.Scatter(
                    x=[float(X[cursor_idx, 0])],
                    y=[float(X[cursor_idx, 1])],
                    mode="markers",
                    showlegend=False,
                )
            )
    else:
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="lines", name="(θ₁, ω₁)"))
        fig.add_trace(go.Scatter(x=X[:, 2], y=X[:, 3], mode="lines", name="(θ₂, ω₂)"))

        if cursor_idx is not None:
            fig.add_trace(go.Scatter(x=[float(X[cursor_idx, 0])], y=[float(X[cursor_idx, 1])], mode="markers", showlegend=False))
            fig.add_trace(go.Scatter(x=[float(X[cursor_idx, 2])], y=[float(X[cursor_idx, 3])], mode="markers", showlegend=False))


    fig.update_layout(height=280, margin=dict(l=10, r=10, t=25, b=10))
    return fig


def make_energy(T: np.ndarray, KE: np.ndarray, PE: np.ndarray, E: np.ndarray, cursor_idx: Optional[int] = None) -> go.Figure:
    fig = go.Figure()
    cursor_idx = _clamp_idx(cursor_idx, len(T))

    fig.add_trace(go.Scatter(x=T, y=KE, mode="lines", name="T kinetic"))
    fig.add_trace(go.Scatter(x=T, y=PE, mode="lines", name="V potential"))
    fig.add_trace(go.Scatter(x=T, y=E, mode="lines", name="E total"))

    if cursor_idx is not None:
        t = float(T[cursor_idx])
        fig.add_vline(x=t, line_width=1, line_dash="dash")
        fig.add_trace(go.Scatter(x=[t], y=[float(E[cursor_idx])], mode="markers", showlegend=False))


    fig.update_layout(xaxis_title="t [s]", yaxis_title="Energy [J]", height=280, margin=dict(l=10, r=10, t=25, b=10))
    return fig
