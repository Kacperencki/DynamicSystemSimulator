# apps/streamlit/vanderpol_view.py

import sys
from pathlib import Path
from typing import Dict, Tuple

import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dss.models import get_model   # VanDerPol under key "vanderpol" :contentReference[oaicite:4]{index=4}
from dss.core.solver import Solver


def _run_vdp(params: Dict, ic: Dict, t0: float, t1: float, dt: float) -> Tuple[Dict, Dict]:
    L, C, mu = params["L"], params["C"], params["mu"]
    v0, iL0 = ic["v0"], ic["iL0"]

    T_total = float(t1 - t0)
    fps_eff = max(1, int(round(1.0 / dt))) if dt > 0 else 100

    system = get_model("vanderpol", mode="default", L=L, C=C, mu=mu)
    sol = Solver(system, initial_conditions=[v0, iL0], T=T_total, fps=fps_eff).run()

    T = sol.t
    X = sol.y.T

    cfg = dict(
        sys="vanderpol",
        L=L,
        C=C,
        mu=mu,
        v0=v0,
        iL0=iL0,
        t0=t0,
        t1=t1,
        dt=dt,
    )
    out = dict(T=T, X=X)
    return cfg, out


def render_vanderpol_page():
    for k, v in {"vdp_out": None, "vdp_cfg": None, "vdp_run": False}.items():
        st.session_state.setdefault(k, v)

    with st.sidebar:
        st.subheader("Van der Pol parameters")

        L = st.number_input("Inductance L [H]", value=1.0, min_value=0.0)
        C = st.number_input("Capacitance C [F]", value=1.0, min_value=0.0)
        mu = st.number_input("Nonlinearity μ", value=1.0)

        with st.expander("Simulation time", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                t0 = st.number_input("t₀ [s]", value=0.0)
            with col2:
                t1 = st.number_input("t₁ [s]", value=40.0)
            with col3:
                dt = st.number_input(
                    "Δt [s]",
                    value=0.01,
                    min_value=1e-4,
                    step=0.001,
                    format="%.4f",
                )

        with st.expander("Initial conditions", expanded=False):
            v0 = st.number_input("v(0) [V]", value=2.0)
            iL0 = st.number_input("iL(0) [A]", value=0.0)

        run_btn = st.button("Run Van der Pol", type="primary")

    if run_btn:
        st.session_state["vdp_run"] = True

    if st.session_state["vdp_run"]:
        params = dict(L=L, C=C, mu=mu)
        ic = dict(v0=v0, iL0=iL0)
        cfg, out = _run_vdp(params, ic, t0, t1, dt)
        st.session_state["vdp_cfg"] = cfg
        st.session_state["vdp_out"] = out

    out = st.session_state["vdp_out"]
    cfg = st.session_state["vdp_cfg"]

    if out is None or cfg is None:
        st.info("Set parameters and click **Run Van der Pol** to start.")
        return

    T, X = out["T"], out["X"]
    v = X[:, 0]
    iL = X[:, 1]

    st.subheader("Van der Pol oscillator")

    top1, top2 = st.columns(2)
    with top1:
        st.markdown("**Phase portrait (v, iL)**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=v, y=iL, mode="lines", name="trajectory"))
        fig.update_layout(
            xaxis_title="v [V]",
            yaxis_title="iL [A]",
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with top2:
        st.markdown("**v(t) and iL(t)**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=T, y=v, mode="lines", name="v(t)"))
        fig.add_trace(go.Scatter(x=T, y=iL, mode="lines", name="iL(t)"))
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="t [s]",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Configuration snapshot"):
        st.code(str(cfg))
