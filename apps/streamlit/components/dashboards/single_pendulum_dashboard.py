from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.streamlit.components.dashboards._common import downsample_idx, pad_range
Cfg = Dict[str, Any]
Out = Dict[str, Any]



def _tip_xy_vectorized(L: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    px = L * np.sin(theta)
    py = -L * np.cos(theta)
    return px.astype(float), py.astype(float)



def make_single_pendulum_dashboard(cfg: Cfg, out: Out, ui: Dict[str, Any]) -> go.Figure:
    T: np.ndarray = out["T"]
    X: np.ndarray = out["X"]
    theta = X[:, 0]
    omega = X[:, 1]

    L = float(cfg["L"])
    px, py = _tip_xy_vectorized(L, theta)

    fps_anim = int(ui.get("fps_anim", 60))
    max_frames = int(ui.get("max_frames", 360))
    max_plot_pts = int(ui.get("max_plot_pts", 2000))
    trail_on = bool(ui.get("trail_on", False))
    trail_max_points = int(ui.get("trail_max_points", 180))

    dt_sim = float(np.mean(np.diff(T))) if len(T) > 1 else max(float(cfg.get("dt", 0.01)), 1e-6)
    step = max(1, int(round(1.0 / (fps_anim * dt_sim))))
    frame_idx = np.arange(0, len(T), dtype=int)[::step]
    if len(frame_idx) > max_frames:
        pick = np.linspace(0, len(frame_idx) - 1, max_frames, dtype=int)
        frame_idx = frame_idx[pick]

    duration_ms = int(round(1000.0 / max(1, fps_anim)))
    pidx = downsample_idx(len(T), max_plot_pts)

    rng = 1.15 * L
    th_min, th_max = pad_range(theta)
    w_min, w_max = pad_range(omega)
    t_min = float(T[0]) if len(T) else 0.0
    t_max = float(T[-1]) if len(T) else 1.0

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"rowspan": 3}, {}], [None, {}], [None, {}]],
        column_widths=[0.64, 0.36],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
        subplot_titles=("", "", "", ""),
    )

    i0 = int(frame_idx[0]) if len(frame_idx) else 0

    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, hoverinfo="skip", name="trail"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0.0, float(px[i0])], y=[0.0, float(py[i0])], mode="lines", showlegend=False, name="rod"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[float(px[i0])], y=[float(py[i0])], mode="markers", marker=dict(size=9), showlegend=False, name="bob"), row=1, col=1)

    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="θ live", line=dict(width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="θ marker", marker=dict(size=6)), row=1, col=2)

    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="ω live", line=dict(width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="ω marker", marker=dict(size=6)), row=2, col=2)

    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="phase live", line=dict(width=2)), row=3, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="phase marker", marker=dict(size=6)), row=3, col=2)

    fig.update_xaxes(range=[-rng, rng], row=1, col=1, showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-rng, rng], row=1, col=1, showgrid=False, zeroline=False, visible=False, scaleanchor="x")

    fig.update_xaxes(range=[t_min, t_max], row=1, col=2, autorange=False, fixedrange=True)
    fig.update_yaxes(range=[th_min, th_max], row=1, col=2, autorange=False, fixedrange=True)

    fig.update_xaxes(range=[t_min, t_max], row=2, col=2, autorange=False, fixedrange=True)
    fig.update_yaxes(range=[w_min, w_max], row=2, col=2, autorange=False, fixedrange=True)

    fig.update_xaxes(range=[th_min, th_max], row=3, col=2, autorange=False, fixedrange=True, title_text="θ")
    fig.update_yaxes(range=[w_min, w_max], row=3, col=2, autorange=False, fixedrange=True, title_text="ω")
    fig.update_xaxes(title_text="t [s]", row=2, col=2)

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

        if trail_on:
            i0_tr = max(0, i - 5000)
            xs = px[i0_tr:i + 1]
            ys = py[i0_tr:i + 1]
            if len(xs) > trail_max_points:
                stride = int(np.ceil(len(xs) / trail_max_points))
                xs = xs[::stride]
                ys = ys[::stride]
            trail_x, trail_y = xs, ys
        else:
            trail_x, trail_y = [], []

        j = int(np.searchsorted(pidx, i, side="right")) - 1
        if j < 0:
            th_x = th_y = []
            w_x = w_y = []
            ph_x = ph_y = []
            th_mx = th_my = []
            w_mx = w_my = []
            ph_mx = ph_my = []
        else:
            sel = pidx[: j + 1]
            th_x = T[sel]
            th_y = theta[sel]
            w_x = T[sel]
            w_y = omega[sel]
            ph_x = theta[sel]
            ph_y = omega[sel]

            th_mx = [float(th_x[-1])]
            th_my = [float(th_y[-1])]
            w_mx = [float(w_x[-1])]
            w_my = [float(w_y[-1])]
            ph_mx = [float(ph_x[-1])]
            ph_my = [float(ph_y[-1])]

        frames.append(
            go.Frame(
                name=f"f{i}",
                data=[
                    go.Scatter(x=trail_x, y=trail_y),                                # 0
                    go.Scatter(x=[0.0, float(px[i])], y=[0.0, float(py[i])]),         # 1
                    go.Scatter(x=[float(px[i])], y=[float(py[i])]),                   # 2
                    go.Scatter(x=th_x, y=th_y),                                       # 3
                    go.Scatter(x=th_mx, y=th_my),                                     # 4
                    go.Scatter(x=w_x, y=w_y),                                         # 5
                    go.Scatter(x=w_mx, y=w_my),                                       # 6
                    go.Scatter(x=ph_x, y=ph_y),                                       # 7
                    go.Scatter(x=ph_mx, y=ph_my),                                     # 8
                ],
                traces=[0, 1, 2, 3, 4, 5, 6, 7, 8],
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
