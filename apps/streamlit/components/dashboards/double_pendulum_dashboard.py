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
        d = 1.0 if np.isclose(y0, 0.0) else abs(y0) * 0.2
        return y0 - d, y1 + d
    d = 0.08 * (y1 - y0)
    return y0 - d, y1 + d


def _positions_vectorized(l1: float, l2: float, th1: np.ndarray, th2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1 = l1 * np.sin(th1)
    y1 = -l1 * np.cos(th1)
    x2 = x1 + l2 * np.sin(th2)
    y2 = y1 - l2 * np.cos(th2)
    return x1.astype(float), y1.astype(float), x2.astype(float), y2.astype(float)


def make_double_pendulum_dashboard(cfg: Cfg, out: Out, ui: Dict[str, Any]) -> go.Figure:
    T: np.ndarray = out["T"]
    X: np.ndarray = out["X"]

    th1 = X[:, 0]
    w1 = X[:, 1]
    th2 = X[:, 2]
    w2 = X[:, 3]

    l1 = float(cfg["l1"])
    l2 = float(cfg["l2"])
    x1, y1, x2, y2 = _positions_vectorized(l1, l2, th1, th2)

    fps_anim = int(ui.get("fps_anim", 60))
    max_frames = int(ui.get("max_frames", 360))
    max_plot_pts = int(ui.get("max_plot_pts", 2200))
    trail_on = bool(ui.get("trail_on", False))
    trail_max_points = int(ui.get("trail_max_points", 220))

    dt_sim = float(np.mean(np.diff(T))) if len(T) > 1 else max(float(cfg.get("dt", 0.01)), 1e-6)
    step = max(1, int(round(1.0 / (max(1, fps_anim) * dt_sim))))
    frame_idx = np.arange(0, len(T), dtype=int)[::step]
    if len(frame_idx) > max_frames:
        pick = np.linspace(0, len(frame_idx) - 1, max_frames, dtype=int)
        frame_idx = frame_idx[pick]

    duration_ms = int(round(1000.0 / max(1, fps_anim)))
    pidx = _downsample_idx(len(T), max_plot_pts)

    span = max(l1 + l2, 1e-6)
    rng = 1.15 * float(
        max(
            span,
            np.max(np.abs(x2)) if len(x2) else span,
            np.max(np.abs(y2)) if len(y2) else span,
        )
    )

    th_min, th_max = _pad_range(np.concatenate([th1, th2]) if len(T) else np.array([0.0, 1.0]))
    w_min, w_max = _pad_range(np.concatenate([w1, w2]) if len(T) else np.array([0.0, 1.0]))
    t_min = float(T[0]) if len(T) else 0.0
    t_max = float(T[-1]) if len(T) else 1.0

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{"rowspan": 3}, {}], [None, {}], [None, {}]],
        column_widths=[0.64, 0.36],
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
        subplot_titles=("", "", "", ""),
    )

    i0 = int(frame_idx[0]) if len(frame_idx) else 0

    # --- Animation (left) ---
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, hoverinfo="skip", name="trail"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0.0, float(x1[i0])], y=[0.0, float(y1[i0])], mode="lines", showlegend=False, name="rod1"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[float(x1[i0]), float(x2[i0])], y=[float(y1[i0]), float(y2[i0])], mode="lines", showlegend=False, name="rod2"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[float(x1[i0])], y=[float(y1[i0])], mode="markers", marker=dict(size=8), showlegend=False, name="m1"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[float(x2[i0])], y=[float(y2[i0])], mode="markers", marker=dict(size=10), showlegend=False, name="m2"), row=1, col=1)

    # --- θ(t) (right, top) ---
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="θ1 live", line=dict(width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="θ1 marker", marker=dict(size=6)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="θ2 live", line=dict(width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="θ2 marker", marker=dict(size=6)), row=1, col=2)

    # --- ω(t) (right, middle) ---
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="ω1 live", line=dict(width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="ω1 marker", marker=dict(size=6)), row=2, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="ω2 live", line=dict(width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="ω2 marker", marker=dict(size=6)), row=2, col=2)

    # --- Phase (right, bottom) ---
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="phase1 live", line=dict(width=2)), row=3, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="phase1 marker", marker=dict(size=6)), row=3, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="phase2 live", line=dict(width=2)), row=3, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="phase2 marker", marker=dict(size=6)), row=3, col=2)

    # Axes
    fig.update_xaxes(range=[-rng, rng], row=1, col=1, showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-rng, rng], row=1, col=1, showgrid=False, zeroline=False, visible=False, scaleanchor="x")

    fig.update_xaxes(range=[t_min, t_max], row=1, col=2, autorange=False, fixedrange=True)
    fig.update_yaxes(range=[th_min, th_max], row=1, col=2, autorange=False, fixedrange=True)

    fig.update_xaxes(range=[t_min, t_max], row=2, col=2, autorange=False, fixedrange=True, title_text="t [s]")
    fig.update_yaxes(range=[w_min, w_max], row=2, col=2, autorange=False, fixedrange=True)

    fig.update_xaxes(range=[th_min, th_max], row=3, col=2, autorange=False, fixedrange=True, title_text="θ")
    fig.update_yaxes(range=[w_min, w_max], row=3, col=2, autorange=False, fixedrange=True, title_text="ω")

    fig.update_layout(
        height=480,
        margin=dict(l=6, r=6, t=36, b=6),
        font=dict(size=11),
        showlegend=False,
        hovermode=False,
    )

    if not trail_on:
        fig.data[0].visible = False

    frames = []
    for i in frame_idx:
        i = int(i)

        # Trail for the second mass
        if trail_on:
            i0_tr = max(0, i - 8000)
            xs = x2[i0_tr : i + 1]
            ys = y2[i0_tr : i + 1]
            if len(xs) > trail_max_points:
                stride = int(np.ceil(len(xs) / trail_max_points))
                xs = xs[::stride]
                ys = ys[::stride]
            trail_x, trail_y = xs, ys
        else:
            trail_x, trail_y = [], []

        j = int(np.searchsorted(pidx, i, side="right")) - 1
        if j < 0:
            th1_x = th1_y = []
            th2_x = th2_y = []
            w1_x = w1_y = []
            w2_x = w2_y = []
            ph1_x = ph1_y = []
            ph2_x = ph2_y = []
            th1_mx = th1_my = []
            th2_mx = th2_my = []
            w1_mx = w1_my = []
            w2_mx = w2_my = []
            ph1_mx = ph1_my = []
            ph2_mx = ph2_my = []
        else:
            sel = pidx[: j + 1]
            th1_x, th1_y = T[sel], th1[sel]
            th2_x, th2_y = T[sel], th2[sel]
            w1_x, w1_y = T[sel], w1[sel]
            w2_x, w2_y = T[sel], w2[sel]
            ph1_x, ph1_y = th1[sel], w1[sel]
            ph2_x, ph2_y = th2[sel], w2[sel]

            th1_mx, th1_my = [float(th1_x[-1])], [float(th1_y[-1])]
            th2_mx, th2_my = [float(th2_x[-1])], [float(th2_y[-1])]
            w1_mx, w1_my = [float(w1_x[-1])], [float(w1_y[-1])]
            w2_mx, w2_my = [float(w2_x[-1])], [float(w2_y[-1])]
            ph1_mx, ph1_my = [float(ph1_x[-1])], [float(ph1_y[-1])]
            ph2_mx, ph2_my = [float(ph2_x[-1])], [float(ph2_y[-1])]

        frames.append(
            go.Frame(
                name=f"f{i}",
                data=[
                    go.Scatter(x=trail_x, y=trail_y),
                    go.Scatter(x=[0.0, float(x1[i])], y=[0.0, float(y1[i])]),
                    go.Scatter(x=[float(x1[i]), float(x2[i])], y=[float(y1[i]), float(y2[i])]),
                    go.Scatter(x=[float(x1[i])], y=[float(y1[i])]),
                    go.Scatter(x=[float(x2[i])], y=[float(y2[i])]),
                    go.Scatter(x=th1_x, y=th1_y),
                    go.Scatter(x=th1_mx, y=th1_my),
                    go.Scatter(x=th2_x, y=th2_y),
                    go.Scatter(x=th2_mx, y=th2_my),
                    go.Scatter(x=w1_x, y=w1_y),
                    go.Scatter(x=w1_mx, y=w1_my),
                    go.Scatter(x=w2_x, y=w2_y),
                    go.Scatter(x=w2_mx, y=w2_my),
                    go.Scatter(x=ph1_x, y=ph1_y),
                    go.Scatter(x=ph1_mx, y=ph1_my),
                    go.Scatter(x=ph2_x, y=ph2_y),
                    go.Scatter(x=ph2_mx, y=ph2_my),
                ],
                traces=list(range(17)),
            )
        )

    fig.frames = frames

    if frames:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.01,
                    y=1.10,
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
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="Reset",
                            method="animate",
                            args=[
                                [frames[0].name],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ]
        )

    return fig
