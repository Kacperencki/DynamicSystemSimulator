# apps/streamlit/views/single_pendulum_view.py

from typing import Dict, Any, Tuple
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dss.models.pendulum import Pendulum
from apps.streamlit.runners.pendulum_runner import run_single_pendulum
from apps.streamlit.components.plots_view import (
    make_time_series,
    make_phase_space,
    make_energy,
)


Cfg = Dict[str, Any]
Out = Dict[str, Any]


def _controls_sidebar() -> Dict[str, Any]:
    """All single-pendulum controls, packed into the sidebar."""
    st.subheader("Single pendulum")

    # --- Mode ---
    mode = st.selectbox("Mode", ["ideal", "damped", "driven", "dc_driven"], index=0)
    is_damped = mode in ["damped", "driven", "dc_driven"]
    is_driven = mode == "driven"

    # --- Physical parameters (expander) ---
    with st.expander("Physical parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            L = st.number_input("L [m]", value=1.0, min_value=0.0)
        with c2:
            m = st.number_input("m [kg]", value=1.0, min_value=0.0)
        with c3:
            g = st.number_input("g [m/s²]", value=9.81)

    # --- Damping / drive (expander) ---
    with st.expander("Damping / drive", expanded=is_damped or is_driven):
        if is_damped:
            db1, db2 = st.columns(2)
            with db1:
                b = st.number_input("b [N·m·s]", value=0.02, min_value=0.0)
            with db2:
                fc = st.number_input(
                    "Coulomb friction F_c [N·m]",
                    value=0.0,
                    min_value=0.0,
                )
        else:
            b = 0.0
            fc = 0.0

        if is_driven:
            d1, d2, d3 = st.columns(3)
            with d1:
                A = st.number_input("A [N·m]", value=0.2, min_value=0.0)
            with d2:
                w = st.number_input("ω [rad/s]", value=2.0, min_value=0.0)
            with d3:
                phi = st.number_input("φ [rad]", value=0.0)
        else:
            A = w = phi = 0.0

    # --- Initial state (expander, after parameters) ---
    with st.expander("Initial state", expanded=True):
        th0 = st.slider(
            "θ₀ [rad]", -float(np.pi), float(np.pi), float(np.pi / 3), 0.01
        )
        w0 = st.slider(
            "ω₀ [rad/s]", -10.0, 10.0, 0.0, 0.1
        )

    # --- Simulation time (ALWAYS at bottom) ---
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

    # --- Run button ---
    run_clicked = st.button(
        "Run single pendulum", type="primary", use_container_width=True
    )

    return dict(
        mode=mode,
        L=L,
        m=m,
        g=g,
        b=b,
        fc=fc,
        A=A,
        w=w,
        phi=phi,
        t0=t0,
        t1=t1,
        dt=dt,
        theta0=th0,
        omega0=w0,
        run_clicked=run_clicked,
    )


def _run_sim(controls: Dict[str, Any]) -> Tuple[Cfg, Out]:
    params = dict(
        mode=controls["mode"],
        L=controls["L"],
        m=controls["m"],
        g=controls["g"],
        b=controls["b"],
        fc=controls["fc"],
        A=controls["A"],
        w=controls["w"],
        phi=controls["phi"],
    )
    ic = dict(theta0=controls["theta0"], omega0=controls["omega0"])

    return run_single_pendulum(params, ic, controls["t0"], controls["t1"], controls["dt"])


def _make_animation(cfg: Cfg, T: np.ndarray, X: np.ndarray) -> go.Figure:
    pend = Pendulum(
        length=cfg["L"],
        mass=cfg["m"],
        mode=cfg["mode"],
        damping=cfg["b"],
        drive_amplitude=cfg["A"],
        drive_frequency=cfg["w"],
        drive_phase=cfg["phi"],
        gravity=cfg["g"],
    )
    tips = np.array([pend.positions(s)[-1] for s in X])
    px, py = tips[:, 0], tips[:, 1]

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
                    go.Scatter(x=[0, px[i]], y=[0, py[i]], mode="lines"),
                    go.Scatter(
                        x=[px[i]], y=[py[i]], mode="markers", marker=dict(size=12)
                    ),
                ],
            )
        )

    k0 = idx[0]
    rng = 1.2 * float(max(cfg["L"], np.max(np.abs(px)), np.max(np.abs(py))))
    fig = go.Figure(
        data=[
            go.Scatter(x=[0, px[k0]], y=[0, py[k0]], mode="lines", name="rod"),
            go.Scatter(
                x=[px[k0]], y=[py[k0]], mode="markers", name="bob", marker=dict(size=12)
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
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return fig


def render_single_pendulum_page() -> None:
    # initialise session state keys
    for k, v in {"sp_cfg": None, "sp_out": None}.items():
        st.session_state.setdefault(k, v)

    # ----- LEFT: controls (sidebar) -----
    with st.sidebar:
        controls = _controls_sidebar()

    run_clicked = bool(controls.pop("run_clicked", False))

    if run_clicked:
        cfg, out = _run_sim(controls)
        st.session_state["sp_cfg"] = cfg
        st.session_state["sp_out"] = out

    cfg = st.session_state["sp_cfg"]
    out = st.session_state["sp_out"]

    # ----- RIGHT: results -----
    st.subheader("Single pendulum")

    if cfg is None or out is None:
        st.info("Set parameters in the left panel and run the simulation.")
        return

    T, X = out["T"], out["X"]
    KE, PE, E = out["E_parts"]

    # No repeated "Mode" here – only timing info
    st.caption(
        f"t ∈ [{cfg['t0']:.1f}, {cfg['t1']:.1f}] s · "
        f"Δt ≈ {cfg['dt']:.3f} s"
    )

    fig_anim = _make_animation(cfg, T, X)
    st.plotly_chart(fig_anim, use_container_width=True)

    tab_time, tab_phase, tab_energy = st.tabs(["Time series", "Phase space", "Energy"])
    with tab_time:
        st.plotly_chart(make_time_series(cfg, T, X), use_container_width=True)
    with tab_phase:
        st.plotly_chart(make_phase_space(cfg, X), use_container_width=True)
    with tab_energy:
        st.plotly_chart(make_energy(T, KE, PE, E), use_container_width=True)
