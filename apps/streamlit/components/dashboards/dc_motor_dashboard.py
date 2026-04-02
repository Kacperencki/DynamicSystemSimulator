from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.streamlit.components.dashboards._common import downsample_idx, pad_range, cfg_param, solver_param, duration_ms_from_frames, animation_buttons
Cfg = Dict[str, Any]
Out = Dict[str, Any]




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
    trail_on = bool(ui.get("trail_on", False))
    trail_max_points = int(ui.get("trail_max_points", 200))

    # frame selection
    dt_sim = float(np.mean(np.diff(T))) if len(T) > 1 else max(float(cfg_param(cfg, "dt", 0.002)), 1e-6)
    step = max(1, int(round(1.0 / (max(1, fps_anim) * dt_sim))))
    frame_idx = np.arange(0, len(T), dtype=int)[::step]
    if len(frame_idx) > max_frames:
        pick = np.linspace(0, len(frame_idx) - 1, max_frames, dtype=int)
        frame_idx = frame_idx[pick]

    duration_ms = duration_ms_from_frames(T, frame_idx, fps_fallback=fps_anim)

    plot_idx = downsample_idx(len(T), max_plot_pts)
    T_p = T[plot_idx]
    i_p = i[plot_idx]
    w_p = omega[plot_idx]
    V_p = V[plot_idx]
    th_p = theta[plot_idx]

    # ranges
    i_min, i_max = pad_range(i)
    w_min, w_max = pad_range(omega)
    V_min, V_max = pad_range(V)
    th_min, th_max = pad_range(theta)
    t_min = float(T[0]) if len(T) else 0.0
    t_max = float(T[-1]) if len(T) else 1.0

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"rowspan": 3}, {}], [None, {}], [None, {}]],
        column_widths=[0.64, 0.36],
        vertical_spacing=0.14,
        horizontal_spacing=0.06,
        subplot_titles=("", "", "", ""),
    )

    # If trail is OFF: show full curves and only move the markers (lighter frames).
    # If trail is ON : show a trailing window of the last `trail_max_points` points.
    if trail_on:
        th_line_x, th_line_y = [], []
        w_line_x, w_line_y = [], []
        i_line_x, i_line_y = [], []
        V_line_x, V_line_y = [], []
    else:
        th_line_x, th_line_y = T_p, th_p
        w_line_x, w_line_y = T_p, w_p
        i_line_x, i_line_y = T_p, i_p
        V_line_x, V_line_y = T_p, V_p

    # Left: theta(t)
    fig.add_trace(go.Scatter(x=th_line_x, y=th_line_y, mode="lines", showlegend=False, name="theta", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[float(T[0])] if len(T) else [], y=[float(theta[0])] if len(theta) else [], mode="markers", showlegend=False, name="theta_m", marker=dict(size=6)), row=1, col=1)

    # Right: omega(t)
    fig.add_trace(go.Scatter(x=w_line_x, y=w_line_y, mode="lines", showlegend=False, name="w", line=dict(width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[float(T[0])] if len(T) else [], y=[float(omega[0])] if len(omega) else [], mode="markers", showlegend=False, name="w_m", marker=dict(size=6)), row=1, col=2)

    # Right: i(t)
    fig.add_trace(go.Scatter(x=i_line_x, y=i_line_y, mode="lines", showlegend=False, name="i", line=dict(width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=[float(T[0])] if len(T) else [], y=[float(i[0])] if len(i) else [], mode="markers", showlegend=False, name="i_m", marker=dict(size=6)), row=2, col=2)

    # Right: V(t)
    fig.add_trace(go.Scatter(x=V_line_x, y=V_line_y, mode="lines", showlegend=False, name="V", line=dict(width=2)), row=3, col=2)
    fig.add_trace(go.Scatter(x=[float(T[0])] if len(T) else [], y=[float(V[0])] if len(V) else [], mode="markers", showlegend=False, name="V_m", marker=dict(size=6)), row=3, col=2)

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
        font=dict(size=10),
        showlegend=False,
        hovermode=False,
    )

    frames = []
    if trail_on:
        # Update both lines (with a trailing window) + markers.
        for k, idx in enumerate(frame_idx):
            idx = int(idx)
            j = int(np.searchsorted(plot_idx, idx, side="right"))
            if j < 1:
                j = 1

            start = max(0, j - max(1, int(trail_max_points)))

            fr = go.Frame(
                name=str(k),
                data=[
                    # theta trail + marker
                    go.Scatter(x=T_p[start:j], y=th_p[start:j]),
                    go.Scatter(x=[float(T[idx])], y=[float(theta[idx])]),
                    # omega trail + marker
                    go.Scatter(x=T_p[start:j], y=w_p[start:j]),
                    go.Scatter(x=[float(T[idx])], y=[float(omega[idx])]),
                    # i trail + marker
                    go.Scatter(x=T_p[start:j], y=i_p[start:j]),
                    go.Scatter(x=[float(T[idx])], y=[float(i[idx])]),
                    # V trail + marker
                    go.Scatter(x=T_p[start:j], y=V_p[start:j]),
                    go.Scatter(x=[float(T[idx])], y=[float(V[idx])]),
                ],
                traces=list(range(8)),
            )
            frames.append(fr)
    else:
        # Only move markers; keep full curves fixed.
        for k, idx in enumerate(frame_idx):
            idx = int(idx)
            fr = go.Frame(
                name=str(k),
                data=[
                    go.Scatter(x=[float(T[idx])], y=[float(theta[idx])]),
                    go.Scatter(x=[float(T[idx])], y=[float(omega[idx])]),
                    go.Scatter(x=[float(T[idx])], y=[float(i[idx])]),
                    go.Scatter(x=[float(T[idx])], y=[float(V[idx])]),
                ],
                traces=[1, 3, 5, 7],
            )
            frames.append(fr)

    fig.frames = frames

    if frames:
        fig.update_layout(updatemenus=animation_buttons(frames, duration_ms, redraw=False, y=1.10))

    return fig
