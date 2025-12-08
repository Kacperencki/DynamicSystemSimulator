# apps/streamlit/inverted_view.py

import sys
from pathlib import Path
from typing import Dict

import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dss.models import get_model
from apps.streamlit.runners.inverted_runner import run_ip_open, run_ip_closed
from apps.streamlit.components.animations import make_cartpole_animation


def render_inverted_page():
    # ensure session keys exist
    for k, v in {"ip_out": None, "ip_cfg": None, "ip_run": False}.items():
        st.session_state.setdefault(k, v)

    # ---- SIDEBAR ----
    with st.sidebar:
        st.subheader("Inverted pendulum")

        # model / mode
        mode = st.selectbox(
            "Mode",
            ["ideal", "damped_cart", "damped_pend", "damped_both", "driven", "dc_driven"],
            index=0,
        )

        st.markdown("### Physical parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            length = st.number_input("Pole length [m]", value=0.3, min_value=0.0)
        with col2:
            mass = st.number_input("Pole mass [kg]", value=0.2, min_value=0.0)
        with col3:
            cart_mass = st.number_input("Cart mass [kg]", value=0.5, min_value=0.0)
        g = st.number_input("Gravity g [m/s²]", value=9.81)

        mass_model = st.selectbox("Mass model", ["point", "uniform"], index=0)

        with st.expander("Simulation time", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                t0 = st.number_input("t₀ [s]", value=0.0)
            with c2:
                t1 = st.number_input("t₁ [s]", value=10.0)
            with c3:
                dt = st.number_input(
                    "Δt [s]",
                    value=0.01,
                    min_value=1e-4,
                    step=0.001,
                    format="%.4f",
                )

        with st.expander("Initial conditions", expanded=False):
            x0 = st.number_input("x(0) [m]", value=0.0)
            xdot0 = st.number_input("ẋ(0) [m/s]", value=0.0)
            th0 = st.number_input("θ(0) [rad]", value=0.05)
            thdot0 = st.number_input("θ̇(0) [rad/s]", value=0.0)

        st.markdown("### Control")

        ctrl_mode = st.selectbox(
            "Control mode",
            [
                "Open-loop (no control)",
                "LQR stabilizer",
                "Swing-up only",
                "Swing-up + LQR (simple)",
            ],
            index=0,
        )

        # parameter dicts for controllers
        lqr_settings: Dict = {}
        swing_settings: Dict = {}
        switch_settings: Dict = {}

        # ---- LQR SETTINGS → SimpleLQR(system, q_x, q_xdot, q_theta, q_thetad, r_u, u_max) ----
        if ctrl_mode in ["LQR stabilizer", "Swing-up + LQR (simple)"]:
            with st.expander("LQR settings", expanded=False):
                q_theta = st.slider("Angle weight q_θ", 10.0, 200.0, 50.0, 5.0)
                q_x = st.slider("Cart position weight q_x", 0.1, 10.0, 1.0, 0.1)
                u_max = st.slider("Max control force |F| [N]", 5.0, 50.0, 20.0, 1.0)

            lqr_settings = dict(
                q_x=q_x,
                q_xdot=0.5,  # fixed but reasonable default
                q_theta=q_theta,
                q_thetad=8.0,
                r_u=0.1,
                u_max=u_max,
            )

        # ---- SWING-UP SETTINGS ----
        if ctrl_mode in ["Swing-up only", "Swing-up + LQR (simple)"]:
            with st.expander("Swing-up settings", expanded=False):
                k_e = st.slider("Energy gain k_e", 0.1, 50.0, 10.0, 0.5)
                F_swing = st.slider("Max swing-up force |F| [N]", 5.0, 80.0, 40.0, 1.0)

            swing_settings = dict(
                k_e=k_e,
                u_max=F_swing,
            )

        # ---- SWITCHER SETTINGS ----
        if ctrl_mode == "Swing-up + LQR (simple)":
            with st.expander("Switcher (Swing-up ↔ LQR)", expanded=False):
                engage_angle = st.slider("Engage LQR below |θ| [deg]", 5.0, 30.0, 12.0, 1.0)
                engage_speed = st.slider("Engage LQR below |θ̇| [rad/s]", 0.5, 10.0, 1.5, 0.5)
                dropout_angle = st.slider("Drop back above |θ| [deg]", 30.0, 90.0, 55.0, 1.0)
                allow_dropout = st.checkbox("Allow dropout back to swing-up", value=True)

            switch_settings = dict(
                engage_angle_deg=engage_angle,
                engage_speed_rad_s=engage_speed,
                dropout_angle_deg=dropout_angle,
                allow_dropout=allow_dropout,
            )

        run_btn = st.button("Run inverted pendulum", type="primary")

    # ---- RUN LOGIC ----
    if run_btn:
        st.session_state["ip_run"] = True

    if st.session_state["ip_run"]:
        params = dict(
            mode=mode,
            length=length,
            mass=mass,
            cart_mass=cart_mass,
            g=g,
            mass_model=mass_model,
        )
        ic = dict(x0=x0, xdot0=xdot0, th0=th0, thdot0=thdot0)

        if ctrl_mode == "Open-loop (no control)":
            cfg, out = run_ip_open(params, ic, t0, t1, dt)
        else:
            cfg, out = run_ip_closed(
                ctrl_mode=ctrl_mode,
                params=params,
                lqr_set=lqr_settings,
                swing_set=swing_settings,
                switch_set=switch_settings,
                ic=ic,
                t0=t0,
                t1=t1,
                dt=dt,
            )

        st.session_state["ip_cfg"] = cfg
        st.session_state["ip_out"] = out

    out = st.session_state["ip_out"]
    cfg = st.session_state["ip_cfg"]

    if out is None or cfg is None:
        st.info("Set parameters and click **Run inverted pendulum** to start.")
        return

    T, X = out["T"], out["X"]
    KE, PE, E = out["E_parts"]

    # ---- MAIN LAYOUT ----
    st.subheader("Cart–pole (inverted pendulum)")

    top_left, top_right = st.columns([3, 1])

    # system object for positions (for animation)
    system = get_model(
        "inverted_pendulum",
        mode=cfg["mode"],
        length=cfg["length"],
        mass=cfg["mass"],
        cart_mass=cfg["cart_mass"],
        gravity=cfg["g"],
        mass_model=cfg["mass_model"],
    )

    with top_left:
        st.markdown("**Cart–pole animation**")
        fig_anim = make_cartpole_animation(cfg, T, X, system)
        st.plotly_chart(fig_anim, use_container_width=True)

    with top_right:
        st.markdown("**Angles & cart position**")
        x = X[:, 0]
        th = X[:, 2]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=T, y=x, mode="lines", name="x(t) [m]"))
        fig.add_trace(go.Scatter(x=T, y=th, mode="lines", name="θ(t) [rad]"))
        fig.update_layout(
            xaxis_title="t [s]",
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Phase space (θ, θ̇)**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[:, 2], y=X[:, 3], mode="lines", name="(θ, θ̇)"))
        fig.update_layout(
            xaxis_title="θ [rad]",
            yaxis_title="θ̇ [rad/s]",
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Energy**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=T, y=KE, mode="lines", name="T"))
        fig.add_trace(go.Scatter(x=T, y=PE, mode="lines", name="V"))
        fig.add_trace(go.Scatter(x=T, y=E, mode="lines", name="E total"))
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Configuration snapshot"):
        st.code(str(cfg))
