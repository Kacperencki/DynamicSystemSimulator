#!/usr/bin/env python3
"""
Generate Chapter 6.2 "diagnostic cards" (white background, black curves, max 2 columns per row).

Outputs (PNG):
  - sp_group.png  : single pendulum (damped) [2x2]
  - dp_group.png  : double pendulum          [2x2]
  - dc_group.png  : DC motor                 [2x2]
  - vdp_group_s.png, vdp_group_m.png, vdp_group_l.png : Van der Pol for 3 mu values [2x2]
  - vdp_phase_overlay_tail.png : overlay of late-time (v, vdot)
  - lz_group.png  : Lorenz system            [3x2]
  - inv_open_group.png  : inverted pendulum open-loop (ideal) [3x2]
  - inv_closed_group.png: inverted pendulum swing-up + LQR (ideal) [3x2]

By default this script IMPORTS YOUR DSS modules and runs simulations via:
    dss.core.simulator.simulate

Usage:
    python tools/ch6_generate_62.py --out figures/chapter_05/section6.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ------------------------- small helpers -------------------------

def _wrap_repo_root_for_import():
    # Allow running from anywhere: add repo root (parent of "dss/") to sys.path.
    here = Path(__file__).resolve()
    # if script lives in tools/, repo root is parent
    cand = here.parent.parent
    if (cand / "dss").exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))


def _save(fig, path: Path, dpi: int = 220):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] wrote {path}")


def _setup_ax(ax):
    # white background, black axes
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    # local wrap to avoid extra imports if needed
    return (theta + np.pi) % (2*np.pi) - np.pi


# ------------------------- generators -------------------------

def gen_pendulum(out: Path, save_csv: bool, T: float, fps: int, method: str, rtol: float, atol: float):
    from dss.models.pendulum import Pendulum
    from dss.core.simulator import simulate

    sys = Pendulum(length=1.0, mass=1.0, mode="damped", damping=0.05)
    y0 = np.array([0.8, 0.0], dtype=float)
    sol, diag = simulate(sys, y0, T=T, fps=fps, method=method, rtol=rtol, atol=atol, return_diagnostics=True)
    t = sol.t
    th, thd = sol.y[0], sol.y[1]

    # energy (if available)
    E = None
    if hasattr(sys, "energy"):
        E = np.array([sys.energy(np.array([th[i], thd[i]])) for i in range(len(t))], dtype=float)
        # normalize if returns vector
        if E.ndim > 1:
            E = E[:, -1]

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.2))
    axs = axs.ravel()

    axs[0].plot(t, th, linewidth=1.2)
    axs[0].set_title(r"$\theta(t)$")
    axs[0].set_xlabel("t [s]"); axs[0].set_ylabel(r"$\theta$ [rad]")
    _setup_ax(axs[0])

    axs[1].plot(t, thd, linewidth=1.2)
    axs[1].set_title(r"$\dot{\theta}(t)$")
    axs[1].set_xlabel("t [s]"); axs[1].set_ylabel(r"$\dot{\theta}$ [rad/s]")
    _setup_ax(axs[1])

    axs[2].plot(th, thd, linewidth=1.0)
    axs[2].set_title(r"Phase portrait $(\theta,\dot{\theta})$")
    axs[2].set_xlabel(r"$\theta$ [rad]"); axs[2].set_ylabel(r"$\dot{\theta}$ [rad/s]")
    _setup_ax(axs[2])

    if E is None:
        axs[3].text(0.5, 0.5, "energy() not available", ha="center", va="center")
    else:
        axs[3].plot(t, E, linewidth=1.2)
    axs[3].set_title(r"Energy diagnostic $E(t)$")
    axs[3].set_xlabel("t [s]"); axs[3].set_ylabel("E [J]")
    _setup_ax(axs[3])

    fig.tight_layout()
    _save(fig, out / "sp_group.png")
    plt.close(fig)

    if save_csv:
        data = {"t": t, "theta": th, "theta_dot": thd}
        if E is not None:
            data["E"] = E
        np.savetxt(out / "sp_timeseries.csv", np.column_stack(list(data.values())), delimiter=",", header=",".join(data.keys()), comments="")


def gen_double_pendulum(out: Path, save_csv: bool, T: float, fps: int, method: str, rtol: float, atol: float):
    from dss.models.double_pendulum import DoublePendulum
    from dss.core.simulator import simulate

    sys = DoublePendulum(length1=1.0, length2=1.0, mass1=1.0, mass2=1.0, mode="ideal")
    y0 = np.array([1.2, 0.0, 1.0, 0.0], dtype=float)  # [th1, th1d, th2, th2d]
    sol, diag = simulate(sys, y0, T=T, fps=fps, method=method, rtol=rtol, atol=atol, return_diagnostics=True)
    t = sol.t
    th1, th1d, th2, th2d = sol.y

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.2))
    axs = axs.ravel()

    axs[0].plot(t, th1, linewidth=1.1, label=r"$\theta_1$")
    axs[0].plot(t, th2, linewidth=1.1, label=r"$\theta_2$")
    axs[0].set_title(r"Angles $\theta_1(t), \theta_2(t)$")
    axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("[rad]")
    axs[0].legend(frameon=False)
    _setup_ax(axs[0])

    axs[1].plot(t, th1d, linewidth=1.1, label=r"$\dot{\theta}_1$")
    axs[1].plot(t, th2d, linewidth=1.1, label=r"$\dot{\theta}_2$")
    axs[1].set_title(r"Angular velocities")
    axs[1].set_xlabel("t [s]"); axs[1].set_ylabel("[rad/s]")
    axs[1].legend(frameon=False)
    _setup_ax(axs[1])

    axs[2].plot(th1, th1d, linewidth=1.0)
    axs[2].set_title(r"Phase portrait $(\theta_1,\dot{\theta}_1)$")
    axs[2].set_xlabel(r"$\theta_1$ [rad]"); axs[2].set_ylabel(r"$\dot{\theta}_1$ [rad/s]")
    _setup_ax(axs[2])

    axs[3].plot(th2, th2d, linewidth=1.0)
    axs[3].set_title(r"Phase portrait $(\theta_2,\dot{\theta}_2)$")
    axs[3].set_xlabel(r"$\theta_2$ [rad]"); axs[3].set_ylabel(r"$\dot{\theta}_2$ [rad/s]")
    _setup_ax(axs[3])

    fig.tight_layout()
    _save(fig, out / "dp_group.png")
    plt.close(fig)

    if save_csv:
        np.savetxt(out / "dp_timeseries.csv",
                   np.column_stack([t, th1, th1d, th2, th2d]),
                   delimiter=",",
                   header="t,theta1,theta1_dot,theta2,theta2_dot",
                   comments="")


def gen_dc_motor(out: Path, save_csv: bool, T: float, fps: int, method: str, rtol: float, atol: float):
    """
    DC motor card: V(t), i(t), omega(t), theta(t).
    Your DCMotor model state is [i, omega]. We compute theta(t) by integrating omega.
    """
    from dss.models.dc_motor import DCMotor
    from dss.core.simulator import simulate

    # Typical demo parameters (stable + clear transients).
    # Replace these with your Chapter 4 parameter set if needed.
    motor = DCMotor(
        R=1.0, L=0.5, Ke=0.1, Kt=0.1,
        J=0.02, bm=0.02,
        v_mode="step", V0=12.0, t_step=0.2,
        load_mode="none",
    )

    y0 = np.array([0.0, 0.0], dtype=float)  # [i, omega]
    sol, diag = simulate(motor, y0, T=T, fps=fps, method=method, rtol=rtol, atol=atol, return_diagnostics=True)
    t = sol.t
    i, w = sol.y

    V = np.array([motor.voltage(float(tt)) for tt in t], dtype=float)

    # theta(t) = ∫ omega dt (cumulative trapezoid)
    theta = np.zeros_like(t)
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        theta[k] = theta[k - 1] + 0.5 * (w[k] + w[k - 1]) * dt

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.2))
    axs = axs.ravel()

    axs[0].plot(t, V, linewidth=1.2, color="black")
    axs[0].set_title(r"Input voltage $V(t)$")
    axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("V [V]")
    _setup_ax(axs[0])

    axs[1].plot(t, i, linewidth=1.2, color="black")
    axs[1].set_title(r"Armature current $i(t)$")
    axs[1].set_xlabel("t [s]"); axs[1].set_ylabel("i [A]")
    _setup_ax(axs[1])

    axs[2].plot(t, w, linewidth=1.2, color="black")
    axs[2].set_title(r"Angular speed $\omega(t)$")
    axs[2].set_xlabel("t [s]"); axs[2].set_ylabel(r"$\omega$ [rad/s]")
    _setup_ax(axs[2])

    axs[3].plot(t, theta, linewidth=1.2, color="black")
    axs[3].set_title(r"Angle $\theta(t)$")
    axs[3].set_xlabel("t [s]"); axs[3].set_ylabel(r"$\theta$ [rad]")
    _setup_ax(axs[3])

    fig.tight_layout()
    _save(fig, out / "dc_group.png")
    plt.close(fig)

    if save_csv:
        np.savetxt(out / "dc_timeseries.csv", np.column_stack([t, V, i, w, theta]),
                   delimiter=",", header="t,V,i,w,theta", comments="")


def gen_vanderpol(out: Path, save_csv: bool, T: float, fps: int, method: str, rtol: float, atol: float):
    """
    Van der Pol oscillator (circuit form in this repo):
      states: v(t), iL(t)
      diagnostics: dv/dt computed from the RHS, and phase portrait (v, dv/dt)

    Outputs:
      - vdp_group_s.png, vdp_group_m.png, vdp_group_l.png
      - vdp_phase_overlay_tail.png (late-time overlay of (v, dv/dt))
    """
    from dss.models import VanDerPol
    from dss.core.simulator import simulate

    mu_vals = [0.5, 2.0, 5.0]
    tags = ["s", "m", "l"]

    tail = []

    for mu, tag in zip(mu_vals, tags):
        sys = VanDerPol(mu=float(mu))
        y0 = np.array([0.1, 0.0], dtype=float)  # [v, iL]
        sol, diag = simulate(sys, y0, T=T, fps=fps, method=method, rtol=rtol, atol=atol, return_diagnostics=True)

        t = sol.t
        v = sol.y[0]
        iL = sol.y[1]

        dvdt = np.zeros_like(t)
        for k in range(len(t)):
            dvdt[k] = float(sys.dynamics(float(t[k]), np.array([v[k], iL[k]], dtype=float))[0])

        # tail (last 25%) for overlay
        k0 = int(0.75 * len(t))
        tail.append((v[k0:], dvdt[k0:], mu))

        fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.2))
        axs = axs.ravel()

        axs[0].plot(v, dvdt, linewidth=1.0)
        axs[0].set_title(r"Phase portrait $(v, \dot v)$")
        axs[0].set_xlabel("v"); axs[0].set_ylabel(r"$\dot v$")
        _setup_ax(axs[0])

        axs[1].plot(t, v, linewidth=1.2)
        axs[1].set_title(r"$v(t)$")
        axs[1].set_xlabel("t [s]"); axs[1].set_ylabel("v")
        _setup_ax(axs[1])

        axs[2].plot(t, iL, linewidth=1.2)
        axs[2].set_title(r"$i_L(t)$")
        axs[2].set_xlabel("t [s]"); axs[2].set_ylabel(r"$i_L$")
        _setup_ax(axs[2])

        axs[3].plot(t, dvdt, linewidth=1.2)
        axs[3].set_title(r"$\dot v(t)$")
        axs[3].set_xlabel("t [s]"); axs[3].set_ylabel(r"$\dot v$")
        _setup_ax(axs[3])

        fig.suptitle(fr"Van der Pol ($\mu={mu}$)", y=1.02)
        fig.tight_layout()
        _save(fig, out / f"vdp_group_{tag}.png")
        plt.close(fig)

        if save_csv:
            np.savetxt(out / f"vdp_timeseries_{tag}.csv", np.column_stack([t, v, iL, dvdt]),
                       delimiter=",", header="t,v,iL,dvdt", comments="")

    # overlay plot
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.2))
    for v_tail, dv_tail, mu in tail:
        ax.plot(v_tail, dv_tail, linewidth=1.0, label=fr"$\mu={mu}$")
    ax.set_title(r"Late-time overlay $(v, \dot v)$")
    ax.set_xlabel("v"); ax.set_ylabel(r"$\dot v$")
    ax.legend(frameon=False)
    _setup_ax(ax)
    fig.tight_layout()
    _save(fig, out / "vdp_phase_overlay_tail.png")
    plt.close(fig)



def gen_lorenz(out: Path, save_csv: bool, T: float, fps: int, method: str, rtol: float, atol: float):
    from dss.models.lorenz import Lorenz
    from dss.core.simulator import simulate

    sys = Lorenz(sigma=10.0, rho=28.0, beta=8/3)
    y0 = np.array([1.0, 1.0, 1.0], dtype=float)
    sol, diag = simulate(sys, y0, T=T, fps=fps, method=method, rtol=rtol, atol=atol, return_diagnostics=True)
    t = sol.t
    x, y, z = sol.y

    fig, axs = plt.subplots(3, 2, figsize=(11.0, 9.0))
    axs = axs.ravel()

    axs[0].plot(t, x, linewidth=1.0); axs[0].set_title("x(t)"); axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("x"); _setup_ax(axs[0])
    axs[1].plot(t, y, linewidth=1.0); axs[1].set_title("y(t)"); axs[1].set_xlabel("t [s]"); axs[1].set_ylabel("y"); _setup_ax(axs[1])
    axs[2].plot(t, z, linewidth=1.0); axs[2].set_title("z(t)"); axs[2].set_xlabel("t [s]"); axs[2].set_ylabel("z"); _setup_ax(axs[2])

    axs[3].plot(x, z, linewidth=0.9); axs[3].set_title("x(z)"); axs[3].set_xlabel("x"); axs[3].set_ylabel("z"); _setup_ax(axs[3])
    axs[4].plot(y, z, linewidth=0.9); axs[4].set_title("y(z)"); axs[4].set_xlabel("y"); axs[4].set_ylabel("z"); _setup_ax(axs[4])
    axs[5].plot(x, y, linewidth=0.9); axs[5].set_title("y(x)"); axs[5].set_xlabel("x"); axs[5].set_ylabel("y"); _setup_ax(axs[5])

    fig.tight_layout()
    _save(fig, out / "lz_group.png")
    plt.close(fig)

    if save_csv:
        np.savetxt(out / "lz_timeseries.csv", np.column_stack([t, x, y, z]),
                   delimiter=",", header="t,x,y,z", comments="")


def gen_inverted(out: Path, save_csv: bool, T: float, fps: int, method: str, rtol: float, atol: float):
    """
    Inverted pendulum (ideal):
      - open-loop: plant alone (u=0)
      - closed-loop: energy swing-up + LQR via SimpleSwitcher
    Initial state: x=0, xdot=0, theta=3.14, thetadot=0
    Params from your screenshot: l=1, M=0.5, m=0.2, g=9.81, coulomb_k=1000.
    """
    from dss.models.inverted_pendulum import InvertedPendulum
    from dss.core.simulator import simulate
    from dss.controllers.swingup import AutoSwingUp
    from dss.controllers.lqr_controller import AutoLQR
    from dss.controllers.simple_switcher import SimpleSwitcher
    from dss.wrappers.closed_loop_cart import ClosedLoopCart

    plant = InvertedPendulum(
        mode="ideal",
        length=1.0,
        cart_mass=0.5,
        mass=0.2,
        gravity=9.81,
        coulomb_k=1000.0,
        # friction terms are ignored in "ideal" mode, but keep them 0
        b_cart=0.0, coulomb_cart=0.0,
        b_pend=0.0, coulomb_pend=0.0,
    )

    y0 = np.array([0.0, 0.0, 3.14, 0.0], dtype=float)

    # --- open-loop
    sol, diag = simulate(plant, y0, T=T, fps=fps, method=method, rtol=rtol, atol=atol, return_diagnostics=True)
    t = sol.t
    x, xd, th, thd = sol.y
    thw = _wrap_to_pi(th)

    fig, axs = plt.subplots(3, 2, figsize=(11.0, 9.0))
    axs = axs.ravel()
    axs[0].plot(t, x, linewidth=1.0, color="black"); axs[0].set_title("x(t)"); axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("x [m]"); _setup_ax(axs[0])
    axs[1].plot(t, thw, linewidth=1.0, color="black"); axs[1].set_title(r"$\theta(t)$ (wrapped)"); axs[1].set_xlabel("t [s]"); axs[1].set_ylabel(r"$\theta$ [rad]"); _setup_ax(axs[1])
    axs[2].plot(t, xd, linewidth=1.0, color="black"); axs[2].set_title(r"$\dot x(t)$"); axs[2].set_xlabel("t [s]"); axs[2].set_ylabel(r"$\dot x$ [m/s]"); _setup_ax(axs[2])
    axs[3].plot(t, thd, linewidth=1.0,  color="black"); axs[3].set_title(r"$\dot\theta(t)$"); axs[3].set_xlabel("t [s]"); axs[3].set_ylabel(r"$\dot\theta$ [rad/s]"); _setup_ax(axs[3])
    axs[4].plot(thw, thd, linewidth=0.9, color="black"); axs[4].set_title(r"Phase $(\theta,\dot\theta)$"); axs[4].set_xlabel(r"$\theta$ [rad]"); axs[4].set_ylabel(r"$\dot\theta$ [rad/s]"); _setup_ax(axs[4])
    axs[5].plot(x, xd, linewidth=0.9,  color="black"); axs[5].set_title(r"Phase $(x,\dot x)$"); axs[5].set_xlabel("x [m]"); axs[5].set_ylabel(r"$\dot x$ [m/s]"); _setup_ax(axs[5])
    fig.suptitle("Inverted pendulum (open-loop, ideal)", y=1.01)
    fig.tight_layout()
    _save(fig, out / "inv_open_group.png")
    plt.close(fig)

    if save_csv:
        np.savetxt(out / "inv_open_timeseries.csv", np.column_stack([t, x, xd, th, thw, thd]),
                   delimiter=",", header="t,x,xdot,theta,theta_wrapped,thetadot", comments="")

    # --- closed-loop: swing-up + LQR with your screenshot tuning
    swing = AutoSwingUp(plant, ke=30.0, force_limit=30.0, du_max=800.0)
    Q = np.diag([10.0, 15.0, 120.0, 250.0])
    lqr = AutoLQR(plant, Q=Q, u_max=30.0)

    ctrl = SimpleSwitcher(
        system=plant,
        lqr_controller=lqr,
        swingup_controller=swing,
        engage_angle_deg=27.0,
        engage_speed_rad_s=9.0,
        engage_cart_speed=6.0,
        dropout_angle_deg=45.0,
        dropout_speed_rad_s=30.0,
        dropout_cart_speed=10.0,
        allow_dropout=True,
    )

    closed = ClosedLoopCart(plant, ctrl)
    sol2, diag2 = simulate(closed, y0, T=T, fps=fps, method=method, rtol=rtol, atol=atol, return_diagnostics=True)
    t2 = sol2.t
    x2, xd2, th2, thd2 = sol2.y
    th2w = _wrap_to_pi(th2)

    fig, axs = plt.subplots(3, 2, figsize=(11.0, 9.0))
    axs = axs.ravel()
    axs[0].plot(t2, x2, linewidth=1.0, color="black"); axs[0].set_title("x(t)"); axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("x [m]"); _setup_ax(axs[0])
    axs[1].plot(t2, th2w, linewidth=1.0, color="black"); axs[1].set_title(r"$\theta(t)$ (wrapped)"); axs[1].set_xlabel("t [s]"); axs[1].set_ylabel(r"$\theta$ [rad]"); _setup_ax(axs[1])
    axs[2].plot(t2, xd2, linewidth=1.0, color="black"); axs[2].set_title(r"$\dot x(t)$"); axs[2].set_xlabel("t [s]"); axs[2].set_ylabel(r"$\dot x$ [m/s]"); _setup_ax(axs[2])
    axs[3].plot(t2, thd2, linewidth=1.0, color="black"); axs[3].set_title(r"$\dot\theta(t)$"); axs[3].set_xlabel("t [s]"); axs[3].set_ylabel(r"$\dot\theta$ [rad/s]"); _setup_ax(axs[3])
    axs[4].plot(th2w, thd2, linewidth=0.9, color="black"); axs[4].set_title(r"Phase $(\theta,\dot\theta)$"); axs[4].set_xlabel(r"$\theta$ [rad]"); axs[4].set_ylabel(r"$\dot\theta$ [rad/s]"); _setup_ax(axs[4])
    axs[5].plot(x2, xd2, linewidth=0.9, color="black"); axs[5].set_title(r"Phase $(x,\dot x)$"); axs[5].set_xlabel("x [m]"); axs[5].set_ylabel(r"$\dot x$ [m/s]"); _setup_ax(axs[5])
    fig.suptitle("Inverted pendulum (swing-up + LQR, ideal)", y=1.01)
    fig.tight_layout()
    _save(fig, out / "inv_closed_group.png")
    plt.close(fig)

    if save_csv:
        # replay u(t) and mode (best-effort)
        # IMPORTANT: controller has internal state; call sequentially in time order
        swing._last_t = None; swing._last_u = 0.0
        ctrl.mode = ctrl.SWING
        u = np.zeros_like(t2)
        mode = []
        for i in range(len(t2)):
            u[i] = float(ctrl.cart_force(float(t2[i]), np.array([x2[i], xd2[i], th2[i], thd2[i]])))
            mode.append(str(ctrl.mode))
        np.savetxt(out / "inv_closed_timeseries.csv",
                   np.column_stack([t2, x2, xd2, th2, th2w, thd2, u]),
                   delimiter=",",
                   header="t,x,xdot,theta,theta_wrapped,thetadot,u",
                   comments="")
        with open(out / "inv_closed_mode.txt", "w", encoding="utf-8") as f:
            for m in mode:
                f.write(m + "\n")


def main():
    _wrap_repo_root_for_import()

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="Output directory for PNGs")
    p.add_argument("--save-csv", action="store_true", help="Also write simple CSV time series")
    p.add_argument("--method", type=str, default="RK45")
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--T", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=200)
    p.add_argument("--lorenz-T", type=float, default=50.0, help="Lorenz horizon (default 50s)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Make matplotlib defaults explicit: black lines on white background
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "black",
        "lines.color": "black",
    })

    params = {
        "solver": {"method": args.method, "rtol": args.rtol, "atol": args.atol},
        "cards": {"T": args.T, "fps": args.fps, "lorenz_T": args.lorenz_T},
        "inverted_screenshot": {
            "l": 1.0, "M": 0.5, "m": 0.2, "g": 9.81, "coulomb_k": 1000.0,
            "y0": [0.0, 0.0, 3.14, 0.0],
            "swing": {"k_e": 30.0, "u_max": 30.0, "du_max": 800.0},
            "lqr": {"Q_diag": [10.0, 15.0, 120.0, 250.0], "u_max": 30.0},
            "switch": {
                "engage_deg": 25.0, "engage_xdot": 6.0, "engage_thetad": 9.0,
                "dropout_deg": 45.0, "dropout_xdot": 10.0, "dropout_thetad": 30.0,
                "allow_dropout": True, "blend_time": 0.12
            }
        }
    }
    (out / "params_used.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    gen_pendulum(out, args.save_csv, args.T, args.fps, args.method, args.rtol, args.atol)
    gen_double_pendulum(out, args.save_csv, args.T, args.fps, args.method, args.rtol, args.atol)
    gen_dc_motor(out, args.save_csv, args.T, args.fps, args.method, args.rtol, args.atol)
    gen_vanderpol(out, args.save_csv, args.T, args.fps, args.method, args.rtol, args.atol)
    gen_lorenz(out, args.save_csv, args.lorenz_T, args.fps, args.method, args.rtol, args.atol)
    gen_inverted(out, args.save_csv, args.T, args.fps, args.method, args.rtol, args.atol)


if __name__ == "__main__":
    main()
