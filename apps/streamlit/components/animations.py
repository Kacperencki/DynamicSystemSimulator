# apps/streamlit/components/animations.py

from typing import Dict

import numpy as np
import plotly.graph_objects as go

from apps.streamlit.components.dashboards._common import duration_ms_from_frames, cfg_param, solver_param


def make_cartpole_animation(cfg: Dict, T, X, system) -> go.Figure:
    """
    Create Plotly animation for cart–pole using the system.positions(state)
    API. Assumes positions return [pivot, tip] with pivot at y=0.
    """
    # compute positions for each state
    pts = [system.positions(s) for s in X]
    cart_x = np.array([p[0][0] for p in pts])
    tip_x = np.array([p[1][0] for p in pts])
    tip_y = np.array([p[1][1] for p in pts])

    if len(T) > 1:
        dt_sim = float(np.mean(np.diff(T)))
    else:
        dt_sim = max(solver_param(cfg, "dt", 0.01), 1e-3)

    fps_anim = 60

    # basic step from dt
    step = max(1, int(round(1.0 / (fps_anim * dt_sim))))
    idx = np.arange(0, len(T), step, dtype=int)

    # hard cap on number of frames (e.g. 400)
    MAX_FRAMES = 400
    if len(idx) > MAX_FRAMES:
        factor = int(np.ceil(len(idx) / MAX_FRAMES))
        idx = idx[::factor]

    if len(idx) == 0:
        idx = np.array([0], dtype=int)

    duration_ms = duration_ms_from_frames(T, idx, fps_fallback=fps_anim)

    cart_w = 0.3
    cart_h = 0.15

    frames = []
    for i in idx:
        cx = cart_x[i]

        cart_shape = go.Scatter(
            x=[
                cx - cart_w / 2,
                cx + cart_w / 2,
                cx + cart_w / 2,
                cx - cart_w / 2,
                cx - cart_w / 2,
            ],
            y=[
                -cart_h / 2,
                -cart_h / 2,
                cart_h / 2,
                cart_h / 2,
                -cart_h / 2,
            ],
            mode="lines",
            line=dict(width=3),
            name="cart",
        )

        pole = go.Scatter(
            x=[cx, tip_x[i]],
            y=[0.0, tip_y[i]],
            mode="lines",
            line=dict(width=4),
            name="pole",
        )

        frames.append(go.Frame(name=f"f{i}", data=[cart_shape, pole]))

    i0 = int(idx[0])
    cx0 = cart_x[i0]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=[
                    cx0 - cart_w / 2,
                    cx0 + cart_w / 2,
                    cx0 + cart_w / 2,
                    cx0 - cart_w / 2,
                    cx0 - cart_w / 2,
                ],
                y=[
                    -cart_h / 2,
                    -cart_h / 2,
                    cart_h / 2,
                    cart_h / 2,
                    -cart_h / 2,
                ],
                mode="lines",
                line=dict(width=3),
                name="cart",
            ),
            go.Scatter(
                x=[cx0, tip_x[i0]],
                y=[0.0, tip_y[i0]],
                mode="lines",
                line=dict(width=4),
                name="pole",
            ),
        ],
        frames=frames,
    )

    span = max(float(np.max(np.abs(cart_x)) + 1.0), float(cfg_param(cfg, "length", 0.5) * 1.5))
    fig.update_layout(
        xaxis=dict(range=[-span, span], scaleanchor="y"),
        yaxis=dict(range=[-cfg_param(cfg, "length", 0.5) - 0.5, cfg_param(cfg, "length", 0.5) + 0.5]),
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "x": 0.05,
                "y": 1.10,
                "xanchor": "left",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": duration_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}}],
                    },
                ],
            }
        ],
    )

    return fig
