from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.streamlit.components.dashboards._common import downsample_idx, pad_range, cfg_param, solver_param, duration_ms_from_frames, animation_buttons
Cfg = Dict[str, Any]
Out = Dict[str, Any]




def make_inverted_pendulum_dashboard(cfg: Cfg, out: Out, ui: Dict[str, Any]) -> go.Figure:
    """
    Build a Plotly figure with a synchronised animation and three time-series plots
    for the inverted pendulum (cart-pole) system.

    Layout: 3-row × 2-column grid.
      Left column (spans all rows): cart-pole animation (cart rectangle + pole line + tip marker).
      Right column row 1: θ(t)  — pole angle vs time.
      Right column row 2: x(t)  — cart position vs time.
      Right column row 3: phase — θ̇ vs θ.

    Parameters
    ----------
    cfg : dict
        Simulation config produced by the runner.  Relevant keys accessed via
        cfg_param()/solver_param() helpers:
          - cfg["model"]["params"]["length"]  pole length [m]  (default 0.3)
          - cfg["solver"]["dt"]               output time step [s]
    out : dict
        Simulation output.  Required keys:
          - "T"  : 1-D array of time points [s]
          - "X"  : 2-D array of states, shape (N, 4), columns [x, ẋ, θ, θ̇]
    ui : dict
        Display settings forwarded from the Streamlit controls.  Keys:
          - "fps_anim"         target animation frame rate (default 60)
          - "max_frames"       hard cap on Plotly animation frames (default 360)
          - "max_plot_pts"     maximum points rendered in the time-series plots (default 2000)
          - "trail_on"         bool — show tip trail in animation (default False)
          - "trail_max_points" length of the tip trail in solver steps (default 180)

    Returns
    -------
    go.Figure
        Plotly figure with animation frames attached, ready for st.plotly_chart().
    """
    T = np.asarray(out["T"], dtype=float)
    X = np.asarray(out["X"], dtype=float)

    x = X[:, 0]
    theta = X[:, 2]
    theta_dot = X[:, 3]

    L = float(cfg_param(cfg, "length", 0.3))

    fps_anim = int(ui.get("fps_anim", 60))
    max_frames = int(ui.get("max_frames", 360))
    max_plot_pts = int(ui.get("max_plot_pts", 2000))
    trail_on = bool(ui.get("trail_on", False))
    trail_max_points = int(ui.get("trail_max_points", 180))

    # Cart geometry in metres (same coordinate system as the simulation).
    # W is at least 0.6 m and scales with pole length so the cart looks proportional.
    # H is 35 % of W; pivot_y places the pole hinge at the top face of the cart.
    W = max(0.6, 2.0 * L)         # cart width  [m]
    H = 0.35 * W                  # cart height [m]
    pivot_y = H                   # y-coordinate of the pole pivot (= cart top face)

    tip_x = x + L * np.sin(theta)
    tip_y = pivot_y + L * np.cos(theta)

    # animation frame selection
    dt_sim = float(np.mean(np.diff(T))) if len(T) > 1 else max(float(solver_param(cfg, "dt", 0.01)), 1e-6)
    step = max(1, int(round(1.0 / (fps_anim * dt_sim))))
    frame_idx = np.arange(0, len(T), dtype=int)[::step]
    if len(frame_idx) > max_frames:
        pick = np.linspace(0, len(frame_idx) - 1, max_frames, dtype=int)
        frame_idx = frame_idx[pick]

    duration_ms = duration_ms_from_frames(T, frame_idx, fps_fallback=fps_anim)

    plot_idx = downsample_idx(len(T), max_plot_pts)
    T_p = T[plot_idx]
    x_p = x[plot_idx]
    th_p = theta[plot_idx]
    thd_p = theta_dot[plot_idx]

    # axis ranges
    x_min = float(min(np.min(x - 0.5 * W), np.min(tip_x))) if len(x) else -1.0
    x_max = float(max(np.max(x + 0.5 * W), np.max(tip_x))) if len(x) else 1.0
    xr_pad = 0.15 * (x_max - x_min if x_max > x_min else 1.0)
    x_min -= xr_pad
    x_max += xr_pad

    y_min = float(min(0.0, np.min(tip_y) if len(tip_y) else 0.0)) - 0.6 * H
    y_max = float(max(pivot_y, np.max(tip_y) if len(tip_y) else pivot_y)) + 0.6 * H

    th_min, th_max = pad_range(theta)
    x_tmin, x_tmax = pad_range(x)
    thd_min, thd_max = pad_range(theta_dot)
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

    i0 = int(frame_idx[0]) if len(frame_idx) else 0

    # --- animation (left) ---
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, hoverinfo="skip", name="trail"), row=1, col=1)
    # cart outline
    cart_x0 = float(x[i0] - 0.5 * W)
    cart_x1 = float(x[i0] + 0.5 * W)
    cart_y0 = 0.0
    cart_y1 = float(H)
    fig.add_trace(
        go.Scatter(
            x=[cart_x0, cart_x1, cart_x1, cart_x0, cart_x0],
            y=[cart_y0, cart_y0, cart_y1, cart_y1, cart_y0],
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            name="cart",
        ),
        row=1, col=1
    )
    # pole
    fig.add_trace(
        go.Scatter(
            x=[float(x[i0]), float(tip_x[i0])],
            y=[pivot_y, float(tip_y[i0])],
            mode="lines",
            showlegend=False,
            name="pole",
        ),
        row=1, col=1
    )
    # tip marker
    fig.add_trace(
        go.Scatter(
            x=[float(tip_x[i0])],
            y=[float(tip_y[i0])],
            mode="markers",
            marker=dict(size=8),
            showlegend=False,
            name="tip",
        ),
        row=1, col=1
    )

    # --- θ(t) (row1 col2) ---
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="θ live", line=dict(width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="θ marker", marker=dict(size=6)), row=1, col=2)

    # --- x(t) (row2 col2) ---
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="x live", line=dict(width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="x marker", marker=dict(size=6)), row=2, col=2)

    # --- phase (row3 col2) ---
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="phase", line=dict(width=2)), row=3, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="phase marker", marker=dict(size=6)), row=3, col=2)

    # layout axes
    fig.update_xaxes(range=[x_min, x_max], showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(range=[y_min, y_max], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, row=1, col=1)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=1, col=2)
    fig.update_yaxes(range=[th_min, th_max], title_text="θ [rad]", row=1, col=2)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=2, col=2)
    fig.update_yaxes(range=[x_tmin, x_tmax], title_text="x [m]", row=2, col=2)

    fig.update_xaxes(range=[th_min, th_max], title_text="θ [rad]", row=3, col=2)
    fig.update_yaxes(range=[thd_min, thd_max], title_text="θ̇ [rad/s]", row=3, col=2)

    frames = []
    for k, i in enumerate(frame_idx):
        i = int(i)
        # plot subset up to i (downsampled)
        j = int(np.searchsorted(plot_idx, i, side="right"))
        if j < 1:
            j = 1

        # trail
        if trail_on:
            s = max(0, i - trail_max_points)
            trail_x = tip_x[s:i+1]
            trail_y = tip_y[s:i+1]
        else:
            trail_x = []
            trail_y = []

        # cart outline points
        cx0 = float(x[i] - 0.5 * W)
        cx1 = float(x[i] + 0.5 * W)
        cy0 = 0.0
        cy1 = float(H)

        fr = go.Frame(
            name=str(k),
            data=[
                # 0 trail
                go.Scatter(x=trail_x, y=trail_y),
                # 1 cart
                go.Scatter(
                    x=[cx0, cx1, cx1, cx0, cx0],
                    y=[cy0, cy0, cy1, cy1, cy0],
                ),
                # 2 pole
                go.Scatter(x=[float(x[i]), float(tip_x[i])], y=[pivot_y, float(tip_y[i])]),
                # 3 tip
                go.Scatter(x=[float(tip_x[i])], y=[float(tip_y[i])]),
                # 4 theta live
                go.Scatter(x=T_p[:j], y=th_p[:j]),
                # 5 theta marker
                go.Scatter(x=[float(T[i])], y=[float(theta[i])]),
                # 6 x live
                go.Scatter(x=T_p[:j], y=x_p[:j]),
                # 7 x marker
                go.Scatter(x=[float(T[i])], y=[float(x[i])]),
                # 8 phase live
                go.Scatter(x=th_p[:j], y=thd_p[:j]),
                # 9 phase marker
                go.Scatter(x=[float(theta[i])], y=[float(theta_dot[i])]),
            ],
            traces=list(range(10)),
        )
        frames.append(fr)

    fig.frames = frames

    if frames:
        fig.update_layout(updatemenus=animation_buttons(frames, duration_ms, redraw=False, y=1.10))

    fig.update_layout(
        height=740,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )

    return fig
