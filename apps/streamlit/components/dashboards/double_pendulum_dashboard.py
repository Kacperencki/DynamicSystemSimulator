from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.streamlit.components.dashboards._common import downsample_idx, pad_range, cfg_param, solver_param, duration_ms_from_frames, animation_buttons
Cfg = Dict[str, Any]
Out = Dict[str, Any]




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

    l1 = float(cfg_param(cfg, "l1", cfg_param(cfg, "L1", 1.0)))
    l2 = float(cfg_param(cfg, "l2", cfg_param(cfg, "L2", 1.0)))
    x1, y1, x2, y2 = _positions_vectorized(l1, l2, th1, th2)

    fps_anim = int(ui.get("fps_anim", 60))
    max_frames = int(ui.get("max_frames", 360))
    max_plot_pts = int(ui.get("max_plot_pts", 2200))
    trail_on = bool(ui.get("trail_on", False))
    trail_max_points = int(ui.get("trail_max_points", 220))
    live_plots = bool(ui.get("live_plots", False))

    dt_sim = float(np.mean(np.diff(T))) if len(T) > 1 else max(float(solver_param(cfg, "dt", 0.01)), 1e-6)
    step = max(1, int(round(1.0 / (max(1, fps_anim) * dt_sim))))
    frame_idx = np.arange(0, len(T), dtype=int)[::step]
    if len(frame_idx) > max_frames:
        pick = np.linspace(0, len(frame_idx) - 1, max_frames, dtype=int)
        frame_idx = frame_idx[pick]

    duration_ms = duration_ms_from_frames(T, frame_idx, fps_fallback=fps_anim)
    pidx = downsample_idx(len(T), max_plot_pts)

    span = max(l1 + l2, 1e-6)
    rng = 1.15 * float(
        max(
            span,
            np.max(np.abs(x2)) if len(x2) else span,
            np.max(np.abs(y2)) if len(y2) else span,
        )
    )

    th_min, th_max = pad_range(np.concatenate([th1, th2]) if len(T) else np.array([0.0, 1.0]))
    w_min, w_max = pad_range(np.concatenate([w1, w2]) if len(T) else np.array([0.0, 1.0]))
    t_min = float(T[0]) if len(T) else 0.0
    t_max = float(T[-1]) if len(T) else 1.0

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{"rowspan": 3}, {}], [None, {}], [None, {}]],
        column_widths=[0.64, 0.36],
        vertical_spacing=0.12,
        horizontal_spacing=0.05,
        subplot_titles=("", "", "", ""),
    )

    i0 = int(frame_idx[0]) if len(frame_idx) else 0

    # --- Animation (left) ---
    # Fix colors explicitly so enabling/disabling the trail doesn't change other trace colors.
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            name="trail",
            line=dict(color="#808080", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, float(x1[i0])],
            y=[0.0, float(y1[i0])],
            mode="lines",
            showlegend=False,
            name="rod1",
            line=dict(color="#202020", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[float(x1[i0]), float(x2[i0])],
            y=[float(y1[i0]), float(y2[i0])],
            mode="lines",
            showlegend=False,
            name="rod2",
            line=dict(color="#202020", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[float(x1[i0])],
            y=[float(y1[i0])],
            mode="markers",
            marker=dict(size=8, color="#ff7f0e"),
            showlegend=False,
            name="m1",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[float(x2[i0])],
            y=[float(y2[i0])],
            mode="markers",
            marker=dict(size=10, color="#d62728"),
            showlegend=False,
            name="m2",
        ),
        row=1,
        col=1,
    )

    if not live_plots:
        # Static mode: full lines upfront (not animated), empty marker traces (animated).
        # Trace layout: 0-4=left anim, 5=θ1_line, 6=θ2_line, 7=ω1_line, 8=ω2_line,
        #               9=phase1_line, 10=phase2_line,
        #               11=θ1_m, 12=θ2_m, 13=ω1_m, 14=ω2_m, 15=ph1_m, 16=ph2_m
        fig.add_trace(go.Scatter(x=T[pidx], y=th1[pidx], mode="lines", showlegend=False, name="θ1 line", line=dict(width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=T[pidx], y=th2[pidx], mode="lines", showlegend=False, name="θ2 line", line=dict(width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=T[pidx], y=w1[pidx], mode="lines", showlegend=False, name="ω1 line", line=dict(width=2)), row=2, col=2)
        fig.add_trace(go.Scatter(x=T[pidx], y=w2[pidx], mode="lines", showlegend=False, name="ω2 line", line=dict(width=2)), row=2, col=2)
        fig.add_trace(go.Scatter(x=th1[pidx], y=w1[pidx], mode="lines", showlegend=False, name="phase1 line", line=dict(width=2)), row=3, col=2)
        fig.add_trace(go.Scatter(x=th2[pidx], y=w2[pidx], mode="lines", showlegend=False, name="phase2 line", line=dict(width=2)), row=3, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="θ1 marker", marker=dict(size=6)), row=1, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="θ2 marker", marker=dict(size=6)), row=1, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="ω1 marker", marker=dict(size=6)), row=2, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="ω2 marker", marker=dict(size=6)), row=2, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="phase1 marker", marker=dict(size=6)), row=3, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="phase2 marker", marker=dict(size=6)), row=3, col=2)
        animated_traces = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16]
    else:
        # Live mode: all traces animated cumulatively.
        # Trace layout: 0-4=left anim, 5=θ1_live, 6=θ1_m, 7=θ2_live, 8=θ2_m,
        #               9=ω1_live, 10=ω1_m, 11=ω2_live, 12=ω2_m,
        #               13=ph1_live, 14=ph1_m, 15=ph2_live, 16=ph2_m
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="θ1 live", line=dict(width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="θ1 marker", marker=dict(size=6)), row=1, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="θ2 live", line=dict(width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="θ2 marker", marker=dict(size=6)), row=1, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="ω1 live", line=dict(width=2)), row=2, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="ω1 marker", marker=dict(size=6)), row=2, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="ω2 live", line=dict(width=2)), row=2, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="ω2 marker", marker=dict(size=6)), row=2, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="phase1 live", line=dict(width=2)), row=3, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="phase1 marker", marker=dict(size=6)), row=3, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="phase2 live", line=dict(width=2)), row=3, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="phase2 marker", marker=dict(size=6)), row=3, col=2)
        animated_traces = list(range(17))

    # Axes
    fig.update_xaxes(range=[-rng, rng], row=1, col=1, showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-rng, rng], row=1, col=1, showgrid=False, zeroline=False, visible=False, scaleanchor="x")

    fig.update_xaxes(range=[t_min, t_max], row=1, col=2, autorange=False, fixedrange=True, title_text="t [s]")
    fig.update_yaxes(range=[th_min, th_max], row=1, col=2, autorange=False, fixedrange=True, title_text="θ₁, θ₂ [rad]")

    fig.update_xaxes(range=[t_min, t_max], row=2, col=2, autorange=False, fixedrange=True, title_text="t [s]")
    fig.update_yaxes(range=[w_min, w_max], row=2, col=2, autorange=False, fixedrange=True, title_text="ω₁, ω₂ [rad/s]")

    fig.update_xaxes(range=[th_min, th_max], row=3, col=2, autorange=False, fixedrange=True, title_text="θ [rad]")
    fig.update_yaxes(range=[w_min, w_max], row=3, col=2, autorange=False, fixedrange=True, title_text="ω [rad/s]")

    fig.update_layout(
        height=540,
        margin=dict(l=6, r=6, t=36, b=6),
        font=dict(size=10),
        showlegend=False,
        hovermode=False,
    )

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

        if not live_plots:
            # Static mode: only animate left panel + marker dots.
            frames.append(
                go.Frame(
                    name=f"f{i}",
                    data=[
                        go.Scatter(x=trail_x, y=trail_y),
                        go.Scatter(x=[0.0, float(x1[i])], y=[0.0, float(y1[i])]),
                        go.Scatter(x=[float(x1[i]), float(x2[i])], y=[float(y1[i]), float(y2[i])]),
                        go.Scatter(x=[float(x1[i])], y=[float(y1[i])]),
                        go.Scatter(x=[float(x2[i])], y=[float(y2[i])]),
                        go.Scatter(x=[float(T[i])], y=[float(th1[i])]),   # 11
                        go.Scatter(x=[float(T[i])], y=[float(th2[i])]),   # 12
                        go.Scatter(x=[float(T[i])], y=[float(w1[i])]),    # 13
                        go.Scatter(x=[float(T[i])], y=[float(w2[i])]),    # 14
                        go.Scatter(x=[float(th1[i])], y=[float(w1[i])]),  # 15
                        go.Scatter(x=[float(th2[i])], y=[float(w2[i])]),  # 16
                    ],
                    traces=animated_traces,
                )
            )
        else:
            # Live mode: cumulative data per frame.
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
                    traces=animated_traces,
                )
            )

    fig.frames = frames

    if frames:
        fig.update_layout(updatemenus=animation_buttons(frames, duration_ms, redraw=False, y=1.10))

    return fig
