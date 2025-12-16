from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

Cfg = Dict[str, Any]
Out = Dict[str, Any]


def _downsample_idx(n: int, max_pts: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if max_pts <= 0 or n <= max_pts:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_pts, dtype=int)


def _pad_range(y: np.ndarray) -> Tuple[float, float]:
    y0 = float(np.min(y))
    y1 = float(np.max(y))
    if np.isclose(y0, y1):
        d = 1.0 if np.isclose(y0, y1) and np.isclose(y0, 0.0) else 0.05 * abs(y0)
        return y0 - d, y1 + d
    pad = 0.08 * (y1 - y0)
    return y0 - pad, y1 + pad


def _duration_ms_from_frames(T: np.ndarray, frame_idx: np.ndarray, fps_fallback: int) -> int:
    if len(frame_idx) < 2:
        return int(round(1000.0 / max(1, int(fps_fallback))))
    try:
        dt = float(np.mean(np.diff(T[np.asarray(frame_idx, dtype=int)])))
    except Exception:
        dt = float("nan")
    if not np.isfinite(dt) or dt <= 0:
        return int(round(1000.0 / max(1, int(fps_fallback))))
    return max(1, int(round(1000.0 * dt)))


def make_dc_motor_dashboard(cfg: Cfg, out: Out, ui: Dict[str, Any]) -> go.Figure:
    """DC motor dashboard.

    Main (left) panel: theta(t) – more informative than a rotor glyph animation.
    Right column: omega(t), current i(t), and applied voltage V(t).
    """
    T = np.asarray(out["T"], dtype=float)
    X = np.asarray(out["X"], dtype=float)
    V = np.asarray(out.get("V", np.zeros(len(T))), dtype=float)
    theta = np.asarray(out.get("theta", np.zeros(len(T))), dtype=float)

    i = X[:, 0]
    omega = X[:, 1]

    fps_anim = int(ui.get("fps_anim", 60))
    max_frames = int(ui.get("max_frames", 420))
    max_plot_pts = int(ui.get("max_plot_pts", 2600))

    # frame selection
    dt_sim = float(np.mean(np.diff(T))) if len(T) > 1 else max(float(cfg.get("dt", 0.002)), 1e-6)
    step = max(1, int(round(1.0 / (max(1, fps_anim) * dt_sim))))
    frame_idx = np.arange(0, len(T), dtype=int)[::step]
    if len(frame_idx) > max_frames:
        pick = np.linspace(0, len(frame_idx) - 1, max_frames, dtype=int)
        frame_idx = frame_idx[pick]

    duration_ms = _duration_ms_from_frames(T, frame_idx, fps_anim)

    plot_idx = _downsample_idx(len(T), max_plot_pts)
    T_p = T[plot_idx]
    i_p = i[plot_idx]
    w_p = omega[plot_idx]
    V_p = V[plot_idx]
    th_p = theta[plot_idx]

    # ranges
    i_min, i_max = _pad_range(i)
    w_min, w_max = _pad_range(omega)
    V_min, V_max = _pad_range(V)
    th_min, th_max = _pad_range(theta)
    t_min = float(T[0]) if len(T) else 0.0
    t_max = float(T[-1]) if len(T) else 1.0

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"rowspan": 3}, {}], [None, {}], [None, {}]],
        column_widths=[0.64, 0.36],
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
        subplot_titles=("", "", "", ""),
    )

    # Left: theta(t)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="theta", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="theta_m", marker=dict(size=6)), row=1, col=1)

    # Right: omega(t)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="w", line=dict(width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="w_m", marker=dict(size=6)), row=1, col=2)

    # Right: i(t)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="i", line=dict(width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="i_m", marker=dict(size=6)), row=2, col=2)

    # Right: V(t)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="V", line=dict(width=2)), row=3, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="V_m", marker=dict(size=6)), row=3, col=2)

    # Axes
    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=1, col=1)
    fig.update_yaxes(range=[th_min, th_max], title_text="θ [rad]", row=1, col=1)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=1, col=2)
    fig.update_yaxes(range=[w_min, w_max], title_text="ω [rad/s]", row=1, col=2)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=2, col=2)
    fig.update_yaxes(range=[i_min, i_max], title_text="i [A]", row=2, col=2)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=3, col=2)
    fig.update_yaxes(range=[V_min, V_max], title_text="V [V]", row=3, col=2)

    fig.update_layout(
        height=640,
        margin=dict(l=6, r=6, t=58, b=8),
        font=dict(size=11),
        showlegend=False,
        hovermode=False,
    )

    frames = []
    for k, idx in enumerate(frame_idx):
        idx = int(idx)
        j = int(np.searchsorted(plot_idx, idx, side="right"))
        if j < 1:
            j = 1

        fr = go.Frame(
            name=str(k),
            data=[
                # theta live + marker
                go.Scatter(x=T_p[:j], y=th_p[:j]),
                go.Scatter(x=[float(T[idx])], y=[float(theta[idx])]),
                # omega live + marker
                go.Scatter(x=T_p[:j], y=w_p[:j]),
                go.Scatter(x=[float(T[idx])], y=[float(omega[idx])]),
                # i live + marker
                go.Scatter(x=T_p[:j], y=i_p[:j]),
                go.Scatter(x=[float(T[idx])], y=[float(i[idx])]),
                # V live + marker
                go.Scatter(x=T_p[:j], y=V_p[:j]),
                go.Scatter(x=[float(T[idx])], y=[float(V[idx])]),
            ],
            traces=list(range(8)),
        )
        frames.append(fr)

    fig.frames = frames

    if frames:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.01,
                    y=1.14,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=duration_ms, redraw=False),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")],
                        ),
                        dict(
                            label="Reset",
                            method="animate",
                            args=[[frames[0].name], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")],
                        ),
                    ],
                )
            ]
        )

    return fig
