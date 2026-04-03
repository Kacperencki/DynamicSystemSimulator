from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.streamlit.components.dashboards._common import downsample_idx, pad_range, cfg_param, solver_param, duration_ms_from_frames, animation_buttons
Cfg = Dict[str, Any]
Out = Dict[str, Any]




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
    live_plots = bool(ui.get("live_plots", False))

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
    v_p = v[plot_idx]
    iL_p = iL[plot_idx]
    dv_p = dv_dt[plot_idx]

    # ranges
    v_min, v_max = pad_range(v)
    i_min, i_max = pad_range(iL)
    dv_min, dv_max = pad_range(dv_dt)
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

    if not live_plots:
        # Static mode: trace layout:
        #   0=phase_line (left, static), 1=phase_marker (left, animated),
        #   2=v_line (static), 3=iL_line (static), 4=dv_line (static),
        #   5=v_marker, 6=iL_marker, 7=dv_marker
        if trail_on:
            ph_x_init: Any = []
            ph_y_init: Any = []
        else:
            ph_x_init = v_p
            ph_y_init = dv_p
        fig.add_trace(
            go.Scatter(x=ph_x_init, y=ph_y_init, mode="lines", showlegend=False, name="phase line", line=dict(width=2)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[float(v[i0])],
                y=[float(dv_dt[i0])],
                mode="markers",
                marker=dict(size=8),
                showlegend=False,
                name="marker",
            ),
            row=1, col=1,
        )
        fig.add_trace(go.Scatter(x=T_p, y=v_p, mode="lines", showlegend=False, name="v line", line=dict(width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=T_p, y=iL_p, mode="lines", showlegend=False, name="iL line", line=dict(width=2)), row=2, col=2)
        fig.add_trace(go.Scatter(x=T_p, y=dv_p, mode="lines", showlegend=False, name="dv line", line=dict(width=2)), row=3, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="v marker", marker=dict(size=6)), row=1, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="iL marker", marker=dict(size=6)), row=2, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="dv marker", marker=dict(size=6)), row=3, col=2)
        animated_traces = [1, 5, 6, 7]
        if trail_on:
            animated_traces = [0, 1, 5, 6, 7]
    else:
        # Live mode: all traces animated cumulatively.
        #   0=phase_live (left), 1=phase_marker (left),
        #   2=v_live, 3=v_marker, 4=iL_live, 5=iL_marker, 6=dv_live, 7=dv_marker
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="phase live", line=dict(width=2)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[float(v[i0])],
                y=[float(dv_dt[i0])],
                mode="markers",
                marker=dict(size=8),
                showlegend=False,
                name="marker",
            ),
            row=1, col=1,
        )
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="v live", line=dict(width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="v marker", marker=dict(size=6)), row=1, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="iL live", line=dict(width=2)), row=2, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="iL marker", marker=dict(size=6)), row=2, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False, name="dv live", line=dict(width=2)), row=3, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, name="dv marker", marker=dict(size=6)), row=3, col=2)
        animated_traces = list(range(8))

    # axes
    fig.update_xaxes(range=[v_min, v_max], title_text="v [V]", row=1, col=1)
    fig.update_yaxes(range=[dv_min, dv_max], title_text="dv/dt [V/s]", row=1, col=1)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=1, col=2)
    fig.update_yaxes(range=[v_min, v_max], title_text="v [V]", row=1, col=2)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=2, col=2)
    fig.update_yaxes(range=[i_min, i_max], title_text="iL [A]", row=2, col=2)

    fig.update_xaxes(range=[t_min, t_max], title_text="t [s]", row=3, col=2)
    fig.update_yaxes(range=[dv_min, dv_max], title_text="dv/dt [V/s]", row=3, col=2)

    fig.update_layout(
        height=540,
        margin=dict(l=6, r=6, t=36, b=6),
        font=dict(size=10),
        showlegend=False,
        hovermode=False,
    )

    frames = []
    for k, i in enumerate(frame_idx):
        i = int(i)

        j = int(np.searchsorted(plot_idx, i, side="right"))
        if j < 1:
            j = 1

        if not live_plots:
            # Static mode: animate marker dots only (and phase trail if trail_on).
            if trail_on:
                j0 = max(0, j - trail_max_points)
                ph_x = v_p[j0:j]
                ph_y = dv_p[j0:j]
                fr = go.Frame(
                    name=str(k),
                    data=[
                        go.Scatter(x=ph_x, y=ph_y),                          # 0 phase trail
                        go.Scatter(x=[float(v[i])], y=[float(dv_dt[i])]),    # 1 phase marker
                        go.Scatter(x=[float(T[i])], y=[float(v[i])]),        # 5 v marker
                        go.Scatter(x=[float(T[i])], y=[float(iL[i])]),       # 6 iL marker
                        go.Scatter(x=[float(T[i])], y=[float(dv_dt[i])]),    # 7 dv marker
                    ],
                    traces=animated_traces,
                )
            else:
                fr = go.Frame(
                    name=str(k),
                    data=[
                        go.Scatter(x=[float(v[i])], y=[float(dv_dt[i])]),    # 1 phase marker
                        go.Scatter(x=[float(T[i])], y=[float(v[i])]),        # 5 v marker
                        go.Scatter(x=[float(T[i])], y=[float(iL[i])]),       # 6 iL marker
                        go.Scatter(x=[float(T[i])], y=[float(dv_dt[i])]),    # 7 dv marker
                    ],
                    traces=animated_traces,
                )
        else:
            # Live mode: cumulative data.
            if trail_on:
                j0 = max(0, j - trail_max_points)
                ph_x = v_p[j0:j]
                ph_y = dv_p[j0:j]
            else:
                ph_x = v_p[:j]
                ph_y = dv_p[:j]

            fr = go.Frame(
                name=str(k),
                data=[
                    # phase live
                    go.Scatter(x=ph_x, y=ph_y),
                    # phase marker
                    go.Scatter(x=[float(v[i])], y=[float(dv_dt[i])]),
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
                traces=animated_traces,
            )
        frames.append(fr)

    fig.frames = frames

    if frames:
        fig.update_layout(updatemenus=animation_buttons(frames, duration_ms, redraw=False, y=1.10))

    return fig
