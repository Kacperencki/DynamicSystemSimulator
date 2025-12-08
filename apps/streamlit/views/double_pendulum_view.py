# apps/streamlit/views/double_pendulum_view.py

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dss.models.double_pendulum import DoublePendulum
from apps.streamlit.runners.pendulum_runner import run_double_pendulum
from apps.streamlit.components.plots_view import (
    make_time_series,
    make_phase_space,
    make_energy,
)


Controls = Dict[str, Any]
Cfg = Dict[str, Any]
Out = Dict[str, Any]


def _controls_sidebar() -> Controls:
    st.subheader("Double pendulum")

    # --- Mode ---
    mode = st.selectbox("Mode", ["ideal", "damped", "driven", "dc_driven"], index=0)
    is_damped = mode in ["damped", "driven", "dc_driven"]
    is_driven = mode == "driven"

    # --- Physical parameters (expander) ---
    with st.expander("Physical parameters", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            l1 = st.number_input("l₁ [m]", min_value=0.0, value=1.0)
            m1 = st.number_input("m₁ [kg]", min_value=0.0, value=1.0)
        with c2:
            l2 = st.number_input("l₂ [m]", min_value=0.0, value=1.0)
            m2 = st.number_input("m₂ [kg]", min_value=0.0, value=1.0)

        g = st.number_input("g [m/s²]", value=9.81)

    # --- Damping / drive (expander) ---
    with st.expander("Damping / drive", expanded=is_damped or is_driven):
        if is_damped:
            db1, db2 = st.columns(2)
            with db1:
                b1 = st.number_input("b₁ [N·m·s]", value=0.02, min_value=0.0)
                fc1 = st.number_input(
                    "Coulomb F_c₁ [N·m]", value=0.0, min_value=0.0
                )
            with db2:
                b2 = st.number_input("b₂ [N·m·s]", value=0.02, min_value=0.0)
                fc2 = st.number_input(
                    "Coulomb F_c₂ [N·m]", value=0.0, min_value=0.0
                )
        else:
            b1 = b2 = 0.0
            fc1 = fc2 = 0.0

        if is_driven:
            st.caption("Drive torques")
            d1c, d2c = st.columns(2)
            with d1c:
                A1 = st.number_input("A₁ [N·m]", value=0.1, min_value=0.0)
                w1 = st.number_input("ω₁ [rad/s]", value=2.0, min_value=0.0)
                phi1 = st.number_input("φ₁ [rad]", value=0.0)
            with d2c:
                A2 = st.number_input("A₂ [N·m]", value=0.1, min_value=0.0)
                w2 = st.number_input("ω₂ [rad/s]", value=2.0, min_value=0.0)
                phi2 = st.number_input("φ₂ [rad]", value=0.0)
        else:
            A1 = A2 = 0.0
            w1 = w2 = 0.0
            phi1 = phi2 = 0.0

    # --- Initial state (expander) ---
    with st.expander("Initial state", expanded=True):
        th1_0 = st.slider(
            "θ₁₀ [rad]",
            min_value=-float(np.pi),
            max_value=float(np.pi),
            value=float(np.pi / 2),
            step=0.01,
        )
        th2_0 = st.slider(
            "θ₂₀ [rad]",
            min_value=-float(np.pi),
            max_value=float(np.pi),
            value=float(np.pi / 2 - 0.2),
            step=0.01,
        )
        w1_0 = st.slider(
            "ω₁₀ [rad/s]",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
        )
        w2_0 = st.slider(
            "ω₂₀ [rad/s]",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
        )

    # --- Simulation time (bottom) ---
    with st.expander("Simulation time", expanded=False):
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            t0 = st.number_input("t₀ [s]", value=0.0)
        with tc2:
            t1 = st.number_input("t₁ [s]", value=10.0)
        with tc3:
            dt = st.number_input(
                "Δt [s]",
                value=0.01,
                min_value=1e-4,
                step=0.001,
                format="%.4f",
            )

    run_clicked = st.button(
        "Run double pendulum", type="primary", use_container_width=True
    )

    return dict(
        mode=mode,
        l1=l1,
        m1=m1,
        l2=l2,
        m2=m2,
        g=g,
        t0=t0,
        t1=t1,
        dt=dt,
        b1=b1,
        b2=b2,
        fc1=fc1,
        fc2=fc2,
        A1=A1,
        w1=w1,
        phi1=phi1,
        A2=A2,
        w2=w2,
        phi2=phi2,
        th1_0=th1_0,
        th2_0=th2_0,
        w1_0=w1_0,
        w2_0=w2_0,
        run_clicked=run_clicked,
    )


def _run_sim(controls: Controls) -> Tuple[Cfg, Out]:
    params = dict(
        mode=controls["mode"],
        l1=controls["l1"],
        m1=controls["m1"],
        l2=controls["l2"],
        m2=controls["m2"],
        g=controls["g"],
        b1=controls["b1"],
        b2=controls["b2"],
        fc1=controls["fc1"],
        fc2=controls["fc2"],
        A1=controls["A1"],
        w1=controls["w1"],
        phi1=controls["phi1"],
        A2=controls["A2"],
        w2=controls["w2"],
        phi2=controls["phi2"],
    )
    ic = dict(
        th1_0=controls["th1_0"],
        w1_0=controls["w1_0"],
        th2_0=controls["th2_0"],
        w2_0=controls["w2_0"],
    )

    return run_double_pendulum(params, ic, controls["t0"], controls["t1"], controls["dt"])



def _make_double_animation(cfg: Cfg, T: np.ndarray, X: np.ndarray) -> go.Figure:
    dp = DoublePendulum(
        length1=cfg["l1"],
        mass1=cfg["m1"],
        length2=cfg["l2"],
        mass2=cfg["m2"],
        mode=cfg["mode"],
        gravity=cfg["g"],
        damping1=cfg["b1"],
        damping2=cfg["b2"],
        drive1_amplitude=cfg["A1"],
        drive1_frequency=cfg["w1"],
        drive1_phase=cfg["phi1"],
        drive2_amplitude=cfg["A2"],
        drive2_frequency=cfg["w2"],
        drive2_phase=cfg["phi2"],
        mass_model="uniform",
    )

    pts = [dp.positions(s) for s in X]  # [pivot, m1, m2]
    rod1x = np.array([[p[0][0], p[1][0]] for p in pts])
    rod1y = np.array([[p[0][1], p[1][1]] for p in pts])
    rod2x = np.array([[p[1][0], p[2][0]] for p in pts])
    rod2y = np.array([[p[1][1], p[2][1]] for p in pts])
    m1_pts = np.array([p[1] for p in pts])
    m2_pts = np.array([p[2] for p in pts])

    fps_anim = 60
    dt_sim = float(np.mean(np.diff(T))) if len(T) > 1 else max(cfg["dt"], 1e-3)
    step = max(1, int(round(1.0 / (fps_anim * dt_sim))))
    idx = np.arange(0, len(T), dtype=int)[::step]
    duration_ms = int(1000 / fps_anim)

    frames = []
    for i in idx:
        frames.append(
            go.Frame(
                name=f"f{i}",
                data=[
                    go.Scatter(x=rod1x[i], y=rod1y[i], mode="lines"),
                    go.Scatter(x=rod2x[i], y=rod2y[i], mode="lines"),
                    go.Scatter(
                        x=[m1_pts[i, 0]],
                        y=[m1_pts[i, 1]],
                        mode="markers",
                        marker=dict(size=10),
                    ),
                    go.Scatter(
                        x=[m2_pts[i, 0]],
                        y=[m2_pts[i, 1]],
                        mode="markers",
                        marker=dict(size=12),
                    ),
                ],
            )
        )

    k0 = idx[0]
    span = cfg["l1"] + cfg["l2"]
    rng = 1.2 * float(
        max(
            span,
            np.max(np.abs(m2_pts[:, 0])),
            np.max(np.abs(m2_pts[:, 1])),
        )
    )

    fig = go.Figure(
        data=[
            go.Scatter(x=rod1x[k0], y=rod1y[k0], mode="lines", name="rod1"),
            go.Scatter(x=rod2x[k0], y=rod2y[k0], mode="lines", name="rod2"),
            go.Scatter(
                x=[m1_pts[k0, 0]],
                y=[m1_pts[k0, 1]],
                mode="markers",
                marker=dict(size=10),
                name="m1",
            ),
            go.Scatter(
                x=[m2_pts[k0, 0]],
                y=[m2_pts[k0, 1]],
                mode="markers",
                marker=dict(size=12),
                name="m2",
            ),
        ],
        frames=frames,
    )

    fig.update_layout(
        xaxis=dict(range=[-rng, rng], scaleanchor="y"),
        yaxis=dict(range=[-rng, rng]),
        height=450,
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
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return fig


def render_double_pendulum_page() -> None:
    for k, v in {"dp_cfg": None, "dp_out": None}.items():
        st.session_state.setdefault(k, v)

    # ----- LEFT: controls (sidebar) -----
    with st.sidebar:
        controls = _controls_sidebar()

    run_clicked = bool(controls.pop("run_clicked", False))

    if run_clicked:
        cfg, out = _run_sim(controls)
        st.session_state["dp_cfg"] = cfg
        st.session_state["dp_out"] = out

    cfg = st.session_state["dp_cfg"]
    out = st.session_state["dp_out"]

    # ----- RIGHT: results -----
    st.subheader("Double pendulum")

    if cfg is None or out is None:
        st.info("Set parameters in the left panel and run the simulation.")
        return

    T: np.ndarray = out["T"]
    X: np.ndarray = out["X"]
    KE, PE, E = out["E_parts"]

    # Only timing info; no repeated "Mode"
    st.caption(
        f"t ∈ [{cfg['t0']:.1f}, {cfg['t1']:.1f}] s · "
        f"Δt ≈ {cfg['dt']:.3f} s"
    )

    fig_anim = _make_double_animation(cfg, T, X)
    st.plotly_chart(fig_anim, use_container_width=True)

    tab_time, tab_phase, tab_energy = st.tabs(["Time series", "Phase space", "Energy"])
    with tab_time:
        st.plotly_chart(make_time_series(cfg, T, X), use_container_width=True)
    with tab_phase:
        st.plotly_chart(make_phase_space(cfg, X), use_container_width=True)
    with tab_energy:
        st.plotly_chart(make_energy(T, KE, PE, E), use_container_width=True)
