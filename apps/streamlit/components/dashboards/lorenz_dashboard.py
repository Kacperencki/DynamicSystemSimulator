from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.streamlit.components.dashboards._common import downsample_idx, pad_range, cfg_param, solver_param, duration_ms_from_frames, animation_buttons
Cfg = Dict[str, Any]
Out = Dict[str, Any]




def make_lorenz_dashboard(cfg: Cfg, out: Out, ui: Dict[str, Any]) -> go.Figure:
    T = np.asarray(out["T"], dtype=float)
    X = np.asarray(out["X"], dtype=float)

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    fps_anim = int(ui.get("fps_anim", 60))
    max_frames = int(ui.get("max_frames", 420))
    max_plot_pts = int(ui.get("max_plot_pts", 3000))
    trail_on = bool(ui.get("trail_on", True))
    trail_max_points = int(ui.get("trail_max_points", 350))

    # frame selection
    dt_sim = float(np.mean(np.diff(T))) if len(T) > 1 else max(float(solver_param(cfg, "dt", 0.01)), 1e-6)
    step = max(1, int(round(1.0 / (max(1, fps_anim) * dt_sim))))
    frame_idx = np.arange(0, len(T), dtype=int)[::step]
    if len(frame_idx) > max_frames:
        pick = np.linspace(0, len(frame_idx) - 1, max_frames, dtype=int)
        frame_idx = frame_idx[pick]

    duration_ms = duration_ms_from_frames(T, frame_idx, fps_fallback=fps_anim)

    plot_idx = downsample_idx(len(T), max_plot_pts)
    T_p = T[plot_idx]
    x_p = x[plot_idx]
    y_p = y[plot_idx]
    z_p = z[plot_idx]

    x_min, x_max = pad_range(x)
    y_min, y_max = pad_range(y)
    z_min, z_max = pad_range(z)

    t_min = float(T[0]) if len(T) else 0.0
    t_max = float(T[-1]) if len(T) else 1.0

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "scene", "rowspan": 3}, {}], [None, {}], [None, {}]],
        column_widths=[0.64, 0.36],
        vertical_spacing=0.14,
        horizontal_spacing=0.06,
        subplot_titles=("", "", "", ""),
    )

    i0 = int(frame_idx[0]) if len(frame_idx) else 0
    i1 = int(frame_idx[1]) if len(frame_idx) > 1 else min(i0 + 1, len(T) - 1)

    # --- 3D attractor (left) ---
    # Seed with at least 2 points (Scatter3d lines often render poorly with empty/1-point lines).
    fig.add_trace(
        go.Scatter3d(
            x=[float(x[i0]), float(x[i1])],
            y=[float(y[i0]), float(y[i1])],
            z=[float(z[i0]), float(z[i1])],
            mode="lines",
            showlegend=False,
            name="traj",
            line=dict(width=3),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[float(x[i0])],
            y=[float(y[i0])],
            z=[float(z[i0])],
            mode="markers",
            showlegend=False,
            name="marker",
            marker=dict(size=4),
        ),
        row=1, col=1,
    )

    # --- x(t) ---
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="x", line=dict(width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="x_m", marker=dict(size=6)), row=1, col=2)

    # --- y(t) ---
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="y", line=dict(width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="y_m", marker=dict(size=6)), row=2, col=2)

    # --- z(t) ---
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="z", line=dict(width=2)), row=3, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="z_m", marker=dict(size=6)), row=3, col=2)

    # time axes
    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=1, col=2)
    fig.update_yaxes(range=[x_min, x_max], title_text="x", row=1, col=2)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=2, col=2)
    fig.update_yaxes(range=[y_min, y_max], title_text="y", row=2, col=2)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=3, col=2)
    fig.update_yaxes(range=[z_min, z_max], title_text="z", row=3, col=2)

    # 3D scene
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x", range=[x_min, x_max]),
            yaxis=dict(title="y", range=[y_min, y_max]),
            zaxis=dict(title="z", range=[z_min, z_max]),
            aspectmode="cube",
        ),
        height=660,
        margin=dict(l=6, r=6, t=58, b=8),
        font=dict(size=11),
        showlegend=False,
        hovermode=False,
    )

    frames = []
    for k, i in enumerate(frame_idx):
        i = int(i)
        j = int(np.searchsorted(plot_idx, i, side="right"))
        if j < 2:
            j = min(2, len(plot_idx))

        if trail_on:
            j0 = max(0, j - trail_max_points)
            x_tr = x_p[j0:j]
            y_tr = y_p[j0:j]
            z_tr = z_p[j0:j]
        else:
            x_tr = x_p[:j]
            y_tr = y_p[:j]
            z_tr = z_p[:j]

        # Ensure 3D line has at least 2 points.
        if len(x_tr) < 2:
            jj = min(len(x_p), max(2, j))
            x_tr = x_p[:jj]
            y_tr = y_p[:jj]
            z_tr = z_p[:jj]

        fr = go.Frame(
            name=str(k),
            data=[
                go.Scatter3d(x=x_tr, y=y_tr, z=z_tr),
                go.Scatter3d(x=[float(x[i])], y=[float(y[i])], z=[float(z[i])]),
                go.Scatter(x=T_p[:j], y=x_p[:j]),
                go.Scatter(x=[float(T[i])], y=[float(x[i])]),
                go.Scatter(x=T_p[:j], y=y_p[:j]),
                go.Scatter(x=[float(T[i])], y=[float(y[i])]),
                go.Scatter(x=T_p[:j], y=z_p[:j]),
                go.Scatter(x=[float(T[i])], y=[float(z[i])]),
            ],
            traces=list(range(8)),
        )
        frames.append(fr)

    fig.frames = frames

    if frames:
        fig.update_layout(updatemenus=animation_buttons(frames, duration_ms, redraw=True, y=1.14))

    return fig
