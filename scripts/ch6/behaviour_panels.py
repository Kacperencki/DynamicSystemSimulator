"""Chapter 6.2 — Functional behaviour panels (one script, all models).

This producer generates a consistent set of *publication-ready* figures and CSVs
for the key models presented in Chapter 4, intended to be included in Chapter 6.2.

Outputs (written directly to repo-root/artifacts/ch6/...):
  figs/ch6_behaviour_<model>.png
  data/ch6_behaviour_<model>[...].csv
  logs/ch6_behaviour_manifest.json

The runner executes this script from within the `scripts/` directory.
Therefore, all paths are resolved relative to the repository root.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Imports from DSS (repo root must be importable when run from scripts/)
# -----------------------------------------------------------------------------
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dss.core.experiments import run_simulation_with_diagnostics
from dss.core.logger import SimulationLogger
from dss.models import Pendulum, DoublePendulum, DCMotor, VanDerPol, Lorenz, InvertedPendulum
from dss.controllers import AutoSwingUp, AutoLQR
from dss.wrappers.closed_loop_cart import ClosedLoopCart


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SimCfg:
    T: float
    fps: int
    method: str = "RK45"
    rtol: float = 1e-6
    atol: float = 1e-8


def _ensure_dirs(root: Path) -> Dict[str, Path]:
    figs = root / "figs"
    data = root / "data"
    logs = root / "logs"
    figs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return {"figs": figs, "data": data, "logs": logs}


def _run(system: Any, x0: np.ndarray, cfg: SimCfg, *, logger: SimulationLogger, name: str, meta: Dict[str, Any]):
    sol, diag = run_simulation_with_diagnostics(
        system=system,
        initial_state=x0,
        T=cfg.T,
        fps=cfg.fps,
        method=cfg.method,
        rtol=cfg.rtol,
        atol=cfg.atol,
        logger=logger,
        experiment_name=name,
        extra_meta=meta,
    )
    return sol, diag


def _energy_series(system: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (E, E_minus_E0) if energy_check exists, else (nan, nan)."""
    if not hasattr(system, "energy_check"):
        n = X.shape[1]
        return np.full(n, np.nan), np.full(n, np.nan)

    E_list: List[float] = []
    for k in range(X.shape[1]):
        e = np.asarray(system.energy_check(X[:, k]), dtype=float).ravel()
        E_list.append(float(e[-1]))
    E = np.asarray(E_list, dtype=float)
    return E, (E - E[0])


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Producers per model
# -----------------------------------------------------------------------------


def produce_pendulum(out: Dict[str, Path], logger: SimulationLogger) -> Dict[str, Any]:
    pend = Pendulum(length=1.0, mass=1.0, gravity=9.81, mode="ideal")
    cfg = SimCfg(T=10.0, fps=200, rtol=1e-6, atol=1e-8)
    x0 = np.array([0.5, 0.0], dtype=float)

    sol, diag = _run(pend, x0, cfg, logger=logger, name="ch6_behaviour_pendulum", meta={"chapter": 6, "section": "6.2"})

    t = sol.t
    th = sol.y[0, :]
    thd = sol.y[1, :]
    E, dE = _energy_series(pend, sol.y)

    # tip trajectory
    tip_xy = np.array([pend.positions(sol.y[:, k])[-1] for k in range(sol.y.shape[1])], dtype=float)

    df = pd.DataFrame({"t": t, "theta": th, "theta_dot": thd, "E": E, "E_minus_E0": dE, "x_tip": tip_xy[:, 0], "y_tip": tip_xy[:, 1]})
    df.to_csv(out["data"] / "ch6_behaviour_pendulum.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.0))
    ax[0, 0].plot(t, th)
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel(r"$\theta$ [rad]")
    ax[0, 0].set_title("Angle")
    ax[0, 0].grid(True)

    ax[0, 1].plot(th, thd)
    ax[0, 1].set_xlabel(r"$\theta$ [rad]")
    ax[0, 1].set_ylabel(r"$\dot\theta$ [rad/s]")
    ax[0, 1].set_title("Phase portrait")
    ax[0, 1].grid(True)

    ax[1, 0].plot(t, dE)
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel(r"$E(t) - E(0)$")
    ax[1, 0].set_title("Energy drift")
    ax[1, 0].grid(True)

    ax[1, 1].plot(tip_xy[:, 0], tip_xy[:, 1])
    ax[1, 1].set_xlabel("x [m]")
    ax[1, 1].set_ylabel("y [m]")
    ax[1, 1].set_title("Tip trajectory")
    ax[1, 1].axis("equal")
    ax[1, 1].grid(True)

    _save_fig(fig, out["figs"] / "ch6_behaviour_pendulum.png")

    return {"cfg": asdict(cfg), "diag": diag, "outputs": ["data/ch6_behaviour_pendulum.csv", "figs/ch6_behaviour_pendulum.png"]}


def produce_double_pendulum(out: Dict[str, Path], logger: SimulationLogger) -> Dict[str, Any]:
    dp = DoublePendulum(
        length1=1.0,
        length2=1.0,
        mass1=1.0,
        mass2=1.0,
        gravity=9.81,
        mode="damped",
        damping1=0.02,
        damping2=0.02,
        coulomb1=0.0,
        coulomb2=0.0,
    )

    cfg = SimCfg(T=20.0, fps=200, rtol=1e-6, atol=1e-8)
    x0a = np.array([1.2, 0.0, 1.0, 0.0], dtype=float)
    x0b = np.array([1.2 + 1e-3, 0.0, 1.0, 0.0], dtype=float)

    sol_a, diag_a = _run(dp, x0a, cfg, logger=logger, name="ch6_behaviour_double_pendulum_a", meta={"chapter": 6, "section": "6.2", "case": "A"})
    sol_b, diag_b = _run(dp, x0b, cfg, logger=logger, name="ch6_behaviour_double_pendulum_b", meta={"chapter": 6, "section": "6.2", "case": "B"})

    t = sol_a.t
    th1 = sol_a.y[0, :]
    th1d = sol_a.y[1, :]
    th2 = sol_a.y[2, :]

    # divergence (use wrapped angle difference for theta2)
    dth2 = np.unwrap(sol_a.y[2, :]) - np.unwrap(sol_b.y[2, :])
    dth2_abs = np.abs(dth2)

    E, dE = _energy_series(dp, sol_a.y)

    df = pd.DataFrame({"t": t, "theta1": th1, "theta1_dot": th1d, "theta2": th2, "E": E, "E_minus_E0": dE, "abs_delta_theta2": dth2_abs})
    df.to_csv(out["data"] / "ch6_behaviour_double_pendulum.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.0))
    ax[0, 0].plot(t, th1, label=r"$\theta_1$")
    ax[0, 0].plot(t, th2, label=r"$\theta_2$")
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("angle [rad]")
    ax[0, 0].set_title("Angles")
    ax[0, 0].grid(True)
    ax[0, 0].legend(loc="best")

    ax[0, 1].plot(th1, th1d)
    ax[0, 1].set_xlabel(r"$\theta_1$ [rad]")
    ax[0, 1].set_ylabel(r"$\dot\theta_1$ [rad/s]")
    ax[0, 1].set_title("Joint-1 phase portrait")
    ax[0, 1].grid(True)

    ax[1, 0].plot(t, dE)
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel(r"$E(t) - E(0)$")
    ax[1, 0].set_title("Energy drift")
    ax[1, 0].grid(True)

    ax[1, 1].plot(t, dth2_abs)
    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel(r"$|\Delta\theta_2|$ [rad]")
    ax[1, 1].set_title("Sensitivity to initial conditions")
    ax[1, 1].grid(True)

    _save_fig(fig, out["figs"] / "ch6_behaviour_double_pendulum.png")

    return {
        "cfg": asdict(cfg),
        "diag": {"A": diag_a, "B": diag_b},
        "outputs": ["data/ch6_behaviour_double_pendulum.csv", "figs/ch6_behaviour_double_pendulum.png"],
        "x0": {"A": x0a.tolist(), "B": x0b.tolist()},
    }


def produce_inverted_pendulum(out: Dict[str, Path], logger: SimulationLogger) -> Dict[str, Any]:
    plant = InvertedPendulum(
        mode="damped_both",
        length=0.3,
        mass=0.2,
        cart_mass=0.5,
        gravity=9.81,
        b_cart=0.1,
        b_pend=0.02,
        coulomb_cart=0.0,
        coulomb_pend=0.0,
    )

    cfg_su = SimCfg(T=8.0, fps=300, rtol=1e-4, atol=1e-6)
    cfg_lqr = SimCfg(T=6.0, fps=300, rtol=1e-4, atol=1e-6)

    swing = AutoSwingUp(system=plant)
    lqr = AutoLQR(system=plant, u_max=20.0)

    closed_su = ClosedLoopCart(system=plant, controller=swing)
    closed_lqr = ClosedLoopCart(system=plant, controller=lqr)

    x0_su = np.array([0.0, 0.0, np.pi - 0.2, 0.0], dtype=float)
    x0_lqr = np.array([0.0, 0.0, 0.15, 0.0], dtype=float)

    sol_su, diag_su = _run(closed_su, x0_su, cfg_su, logger=logger, name="ch6_behaviour_inverted_swingup", meta={"chapter": 6, "section": "6.2", "mode": "swingup"})
    sol_lq, diag_lq = _run(closed_lqr, x0_lqr, cfg_lqr, logger=logger, name="ch6_behaviour_inverted_lqr", meta={"chapter": 6, "section": "6.2", "mode": "lqr"})

    # swing-up series
    t1 = sol_su.t
    x1 = sol_su.y
    th1 = x1[2, :]
    xcart1 = x1[0, :]
    E1, dE1 = _energy_series(plant, x1)

    # LQR series
    t2 = sol_lq.t
    x2 = sol_lq.y
    th2 = x2[2, :]
    xcart2 = x2[0, :]
    E2, dE2 = _energy_series(plant, x2)

    df1 = pd.DataFrame({"t": t1, "x": xcart1, "x_dot": x1[1, :], "theta": th1, "theta_dot": x1[3, :], "E": E1, "E_minus_E0": dE1})
    df2 = pd.DataFrame({"t": t2, "x": xcart2, "x_dot": x2[1, :], "theta": th2, "theta_dot": x2[3, :], "E": E2, "E_minus_E0": dE2})

    df1.to_csv(out["data"] / "ch6_behaviour_inverted_swingup.csv", index=False)
    df2.to_csv(out["data"] / "ch6_behaviour_inverted_lqr.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.0))
    ax[0, 0].plot(t1, th1)
    ax[0, 0].set_title("Swing-up: pendulum angle")
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel(r"$\theta$ [rad]")
    ax[0, 0].grid(True)

    ax[0, 1].plot(t1, xcart1)
    ax[0, 1].set_title("Swing-up: cart position")
    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("x [m]")
    ax[0, 1].grid(True)

    ax[1, 0].plot(t2, th2)
    ax[1, 0].set_title("LQR: pendulum angle")
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel(r"$\theta$ [rad]")
    ax[1, 0].grid(True)

    ax[1, 1].plot(t2, xcart2)
    ax[1, 1].set_title("LQR: cart position")
    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel("x [m]")
    ax[1, 1].grid(True)

    _save_fig(fig, out["figs"] / "ch6_behaviour_inverted_pendulum.png")

    return {
        "cfg": {"swingup": asdict(cfg_su), "lqr": asdict(cfg_lqr)},
        "diag": {"swingup": diag_su, "lqr": diag_lq},
        "outputs": [
            "data/ch6_behaviour_inverted_swingup.csv",
            "data/ch6_behaviour_inverted_lqr.csv",
            "figs/ch6_behaviour_inverted_pendulum.png",
        ],
        "x0": {"swingup": x0_su.tolist(), "lqr": x0_lqr.tolist()},
    }


def produce_dc_motor(out: Dict[str, Path], logger: SimulationLogger) -> Dict[str, Any]:
    motor = DCMotor(
        R=2.0,
        L=0.5,
        Ke=0.06,
        Kt=0.06,
        J=2e-3,
        bm=2e-3,
        v_mode="step",
        V0=6.0,
        v_offset=0.0,
        t_step=0.05,
        load_mode="none",
    )
    cfg = SimCfg(T=2.0, fps=2000, rtol=1e-6, atol=1e-8)
    x0 = np.array([0.0, 0.0], dtype=float)
    sol, diag = _run(motor, x0, cfg, logger=logger, name="ch6_behaviour_dc_motor", meta={"chapter": 6, "section": "6.2"})

    t = sol.t
    i = sol.y[0, :]
    w = sol.y[1, :]
    E, dE = _energy_series(motor, sol.y)
    v = np.array([motor.voltage(float(tt)) for tt in t], dtype=float)

    df = pd.DataFrame({"t": t, "i": i, "omega": w, "v": v, "E": E, "E_minus_E0": dE})
    df.to_csv(out["data"] / "ch6_behaviour_dc_motor.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.0))
    ax[0, 0].plot(t, v)
    ax[0, 0].set_title("Input voltage")
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("v [V]")
    ax[0, 0].grid(True)

    ax[0, 1].plot(t, i)
    ax[0, 1].set_title("Armature current")
    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("i [A]")
    ax[0, 1].grid(True)

    ax[1, 0].plot(t, w)
    ax[1, 0].set_title("Angular speed")
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel(r"$\omega$ [rad/s]")
    ax[1, 0].grid(True)

    ax[1, 1].plot(i, w)
    ax[1, 1].set_title("State trajectory")
    ax[1, 1].set_xlabel("i [A]")
    ax[1, 1].set_ylabel(r"$\omega$ [rad/s]")
    ax[1, 1].grid(True)

    _save_fig(fig, out["figs"] / "ch6_behaviour_dc_motor.png")

    return {"cfg": asdict(cfg), "diag": diag, "outputs": ["data/ch6_behaviour_dc_motor.csv", "figs/ch6_behaviour_dc_motor.png"]}


def produce_vanderpol(out: Dict[str, Path], logger: SimulationLogger) -> Dict[str, Any]:
    vdp = VanDerPol(L=1.0, C=1.0, mu=2.0)
    cfg = SimCfg(T=30.0, fps=200, rtol=1e-6, atol=1e-8)
    x0 = np.array([1.0, 0.0], dtype=float)
    sol, diag = _run(vdp, x0, cfg, logger=logger, name="ch6_behaviour_vanderpol", meta={"chapter": 6, "section": "6.2"})

    t = sol.t
    v = sol.y[0, :]
    iL = sol.y[1, :]

    df = pd.DataFrame({"t": t, "v": v, "iL": iL})
    df.to_csv(out["data"] / "ch6_behaviour_vanderpol.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.0))
    ax[0, 0].plot(t, v)
    ax[0, 0].set_title("Capacitor voltage")
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("v [V]")
    ax[0, 0].grid(True)

    ax[0, 1].plot(t, iL)
    ax[0, 1].set_title("Inductor current")
    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("iL [A]")
    ax[0, 1].grid(True)

    ax[1, 0].plot(v, iL)
    ax[1, 0].set_title("Phase portrait")
    ax[1, 0].set_xlabel("v [V]")
    ax[1, 0].set_ylabel("iL [A]")
    ax[1, 0].grid(True)

    # One-dimensional Poincaré-like cut: show v(t) after transient
    n0 = int(0.25 * t.size)
    ax[1, 1].plot(t[n0:], v[n0:])
    ax[1, 1].set_title("After transient")
    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel("v [V]")
    ax[1, 1].grid(True)

    _save_fig(fig, out["figs"] / "ch6_behaviour_vanderpol.png")

    return {"cfg": asdict(cfg), "diag": diag, "outputs": ["data/ch6_behaviour_vanderpol.csv", "figs/ch6_behaviour_vanderpol.png"]}


def produce_lorenz(out: Dict[str, Path], logger: SimulationLogger) -> Dict[str, Any]:
    lor = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    cfg = SimCfg(T=10.0, fps=200, rtol=1e-6, atol=1e-9)

    x0a = np.array([1.0, 1.0, 1.0], dtype=float)
    x0b = np.array([1.0 + 1e-8, 1.0, 1.0], dtype=float)

    sol_a, diag_a = _run(lor, x0a, cfg, logger=logger, name="ch6_behaviour_lorenz_a", meta={"chapter": 6, "section": "6.2", "case": "A"})
    sol_b, diag_b = _run(lor, x0b, cfg, logger=logger, name="ch6_behaviour_lorenz_b", meta={"chapter": 6, "section": "6.2", "case": "B"})

    t = sol_a.t
    xa, ya, za = sol_a.y[0, :], sol_a.y[1, :], sol_a.y[2, :]
    xb, yb, zb = sol_b.y[0, :], sol_b.y[1, :], sol_b.y[2, :]

    diff = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2)
    diff = np.maximum(diff, 1e-16)

    df = pd.DataFrame({"t": t, "x": xa, "y": ya, "z": za, "delta_norm": diff})
    df.to_csv(out["data"] / "ch6_behaviour_lorenz.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.0))
    ax[0, 0].plot(t, xa)
    ax[0, 0].set_title("x(t)")
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("x")
    ax[0, 0].grid(True)

    ax[0, 1].plot(t, ya)
    ax[0, 1].set_title("y(t)")
    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("y")
    ax[0, 1].grid(True)

    ax[1, 0].plot(xa, ya)
    ax[1, 0].set_title("Projection: (x, y)")
    ax[1, 0].set_xlabel("x")
    ax[1, 0].set_ylabel("y")
    ax[1, 0].grid(True)

    ax[1, 1].plot(t, np.log10(diff))
    ax[1, 1].set_title(r"Divergence: $\log_{10}||\Delta||$")
    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel(r"$\log_{10}||\Delta||$")
    ax[1, 1].grid(True)

    _save_fig(fig, out["figs"] / "ch6_behaviour_lorenz.png")

    return {
        "cfg": asdict(cfg),
        "diag": {"A": diag_a, "B": diag_b},
        "outputs": ["data/ch6_behaviour_lorenz.csv", "figs/ch6_behaviour_lorenz.png"],
        "x0": {"A": x0a.tolist(), "B": x0b.tolist()},
    }


def main() -> None:
    artifacts_root = REPO_ROOT / "artifacts" / "ch6"
    out = _ensure_dirs(artifacts_root)
    logger = SimulationLogger(log_dir=str(out["logs"]))

    results: Dict[str, Any] = {
        "producer": "behaviour_panels",
        "chapter": 6,
        "section": "6.2",
        "outputs_root": str(artifacts_root),
        "models": {},
    }

    results["models"]["pendulum"] = produce_pendulum(out, logger)
    results["models"]["double_pendulum"] = produce_double_pendulum(out, logger)
    results["models"]["inverted_pendulum"] = produce_inverted_pendulum(out, logger)
    results["models"]["dc_motor"] = produce_dc_motor(out, logger)
    results["models"]["vanderpol"] = produce_vanderpol(out, logger)
    results["models"]["lorenz"] = produce_lorenz(out, logger)

    (out["logs"] / "ch6_behaviour_manifest.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
