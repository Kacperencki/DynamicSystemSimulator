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
        d = 1.0 if np.isclose(y0, 0.0) else 0.05 * abs(y0)
        return y0 - d, y1 + d
    pad = 0.08 * (y1 - y0)
    return y0 - pad, y1 + pad


def make_vanderpol_dashboard(cfg: Cfg, out: Out, ui: Dict[str, Any]) -> go.Figure:
    T = np.asarray(out["T"], dtype=float)
    X = np.asarray(out["X"], dtype=float)

    dv_dt = np.asarray(
        out.get(
            "dv_dt",
            np.gradient(X[:, 0], T, edge_order=1) if len(T) > 1 else np.zeros(len(T)),
        ),
        dtype=float,
    )

    v = X[:, 0]
    iL = X[:, 1]

    fps_anim = int(ui.get("fps_anim", 60))
    max_frames = int(ui.get("max_frames", 360))
    max_plot_pts = int(ui.get("max_plot_pts", 2200))
    trail_on = bool(ui.get("trail_on", False))
    trail_max_points = int(ui.get("trail_max_points", 240))

    # frame selection
    dt_sim = float(np.mean(np.diff(T))) if len(T) > 1 else max(float(cfg.get("dt", 0.01)), 1e-6)
    step = max(1, int(round(1.0 / (max(1, fps_anim) * dt_sim))))
    frame_idx = np.arange(0, len(T), dtype=int)[::step]
    if len(frame_idx) > max_frames:
        pick = np.linspace(0, len(frame_idx) - 1, max_frames, dtype=int)
        frame_idx = frame_idx[pick]

    duration_ms = int(round(1000.0 / max(1, fps_anim)))

    plot_idx = _downsample_idx(len(T), max_plot_pts)
    T_p = T[plot_idx]
    v_p = v[plot_idx]
    iL_p = iL[plot_idx]
    dv_p = dv_dt[plot_idx]

    # ranges
    v_min, v_max = _pad_range(v)
    i_min, i_max = _pad_range(iL)
    dv_min, dv_max = _pad_range(dv_dt)
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

    # --- phase portrait (left): dv/dt (x-axis) vs v (y-axis) ---
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="phase live", line=dict(width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[float(dv_dt[i0])],
            y=[float(v[i0])],
            mode="markers",
            marker=dict(size=8),
            showlegend=False,
            name="marker",
        ),
        row=1,
        col=1,
    )

    # --- v(t) ---
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="v live", line=dict(width=2)),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="v marker", marker=dict(size=6)),
        row=1,
        col=2,
    )

    # --- iL(t) ---
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="iL live", line=dict(width=2)),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="iL marker", marker=dict(size=6)),
        row=2,
        col=2,
    )

    # --- dv/dt(t) ---
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="dv live", line=dict(width=2)),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="dv marker", marker=dict(size=6)),
        row=3,
        col=2,
    )

    # axes
    fig.update_xaxes(range=[dv_min, dv_max], title_text="dv/dt [V/s]", row=1, col=1)
    fig.update_yaxes(range=[v_min, v_max], title_text="v [V]", row=1, col=1)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=1, col=2)
    fig.update_yaxes(range=[v_min, v_max], title_text="v [V]", row=1, col=2)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=2, col=2)
    fig.update_yaxes(range=[i_min, i_max], title_text="iL [A]", row=2, col=2)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=3, col=2)
    fig.update_yaxes(range=[dv_min, dv_max], title_text="dv/dt [V/s]", row=3, col=2)

    fig.update_layout(
        height=480,
        margin=dict(l=6, r=6, t=36, b=6),
        font=dict(size=11),
        showlegend=False,
        hovermode=False,
    )

    frames = []
    for k, i in enumerate(frame_idx):
        i = int(i)

        # plot subset end index in downsample grid
        j = int(np.searchsorted(plot_idx, i, side="right"))
        if j < 1:
            j = 1

        if trail_on:
            j0 = max(0, j - trail_max_points)
            ph_x = dv_p[j0:j]
            ph_y = v_p[j0:j]
        else:
            ph_x = dv_p[:j]
            ph_y = v_p[:j]

        fr = go.Frame(
            name=str(k),
            data=[
                # phase live
                go.Scatter(x=ph_x, y=ph_y),
                # phase marker
                go.Scatter(x=[float(dv_dt[i])], y=[float(v[i])]),
                # v(t) live
                go.Scatter(x=T_p[:j], y=v_p[:j]),
                # v marker
                go.Scatter(x=[float(T[i])], y=[float(v[i])]),
                # iL(t) live
                go.Scatter(x=T_p[:j], y=iL_p[:j]),
                # iL marker
                go.Scatter(x=[float(T[i])], y=[float(iL[i])]),
                # dv/dt live
                go.Scatter(x=T_p[:j], y=dv_p[:j]),
                # dv/dt marker
                go.Scatter(x=[float(T[i])], y=[float(dv_dt[i])]),
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
