# scripts/ch6_04_inverted_case_study.py
#
# Inverted pendulum: (1) swing-up from hanging, (2) LQR stabilisation near upright.
# Outputs:
#   data/ch6_inverted_swingup_timeseries.csv
#   data/ch6_inverted_lqr_timeseries.csv
#   data/ch6_inverted_summary.csv
#   figs/ch6_inverted_swingup_theta.png
#   figs/ch6_inverted_lqr_theta.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dss.models import InvertedPendulum
from dss.wrappers.closed_loop_cart import CloseLoopCart
from dss.controllers import AutoLQR, AutoSwingUp
from dss.core.experiments import run_simulation_with_diagnostics
from dss.core.logger import SimulationLogger

from typing import Optional
def estimate_settling_time(t: np.ndarray, y: np.ndarray, eps: float = 0.05) -> Optional[float]:
    """
    Simple settling time: first time after which |y| < eps and stays below.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.abs(y) < eps
    if not mask.any():
        return None

    # last index where |y| >= eps
    idx_last_out = np.where(~mask)[0]
    if idx_last_out.size == 0:
        return float(t[0])
    idx = idx_last_out[-1]
    if idx >= len(t) - 1:
        return None
    return float(t[idx + 1])


def run_swingup(plant: InvertedPendulum, logger: SimulationLogger):
    # Closed loop: plant + swing-up controller
    swing = AutoSwingUp(system=plant)
    closed = CloseLoopCart(system=plant, controller=swing)

    # Start near bottom: theta ≈ π (downward), expressed as angle from UP
    x0 = np.array([0.0, 0.0, np.pi - 0.2, 0.0], dtype=float)
    T_total = 8.0
    fps = 300
    rtol = 1e-4
    atol = 1e-6

    sol, diag = run_simulation_with_diagnostics(
        system=closed,
        initial_state=x0,
        T=T_total,
        fps=fps,
        rtol=rtol,
        atol=atol,
        logger=logger,
        experiment_name="inverted_swingup",
        extra_meta={"chapter": 6, "kind": "inverted_swingup"},
    )

    t = sol.t
    x = sol.y  # [x, x_dot, theta, theta_dot]
    theta = x[2, :]

    # We also compute energy via plant.energy_check(state)
    energies = []
    for k in range(x.shape[1]):
        energies.append(plant.energy_check(x[:, k]))
    energies = np.asarray(energies, dtype=float)
    E_tot = energies[:, -1]
    E_drift = E_tot - E_tot[0]

    df = pd.DataFrame(
        {
            "t": t,
            "x": x[0, :],
            "x_dot": x[1, :],
            "theta": theta,
            "theta_dot": x[3, :],
            "E": E_tot,
            "E_minus_E0": E_drift,
        }
    )
    df.to_csv("data/ch6_inverted_swingup_timeseries.csv", index=False)

    # Plot θ(t)
    plt.figure()
    plt.plot(t, theta)
    plt.xlabel("t [s]")
    plt.ylabel("θ [rad]")
    plt.title("Inverted pendulum: swing-up from hanging")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figs/ch6_inverted_swingup_theta.png", dpi=300)
    plt.close()

    return diag, df


def run_lqr(plant: InvertedPendulum, logger: SimulationLogger):
    # LQR around upright
    lqr = AutoLQR(system=plant, u_max=20.0)
    closed = CloseLoopCart(system=plant, controller=lqr)

    # Small perturbation from upright
    x0 = np.array([0.0, 0.0, 0.15, 0.0], dtype=float)
    T_total = 6.0
    fps = 300
    rtol = 1e-4
    atol = 1e-6

    sol, diag = run_simulation_with_diagnostics(
        system=closed,
        initial_state=x0,
        T=T_total,
        fps=fps,
        rtol=rtol,
        atol=atol,
        logger=logger,
        experiment_name="inverted_lqr",
        extra_meta={"chapter": 6, "kind": "inverted_lqr"},
    )

    t = sol.t
    x = sol.y
    theta = x[2, :]
    x_cart = x[0, :]

    settling_time = estimate_settling_time(t, theta, eps=0.05)

    df = pd.DataFrame(
        {
            "t": t,
            "x": x_cart,
            "x_dot": x[1, :],
            "theta": theta,
            "theta_dot": x[3, :],
        }
    )
    df.to_csv("data/ch6_inverted_lqr_timeseries.csv", index=False)

    # θ(t) plot
    plt.figure()
    plt.plot(t, theta)
    plt.xlabel("t [s]")
    plt.ylabel("θ [rad]")
    plt.title("Inverted pendulum: LQR stabilisation near upright")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figs/ch6_inverted_lqr_theta.png", dpi=300)
    plt.close()

    diag_ext = dict(diag)
    diag_ext["settling_time_theta_eps0.05"] = settling_time
    diag_ext["x_max_abs"] = float(np.max(np.abs(x_cart)))
    return diag_ext, df



def main() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger = SimulationLogger(log_dir="logs")

    # Plant parameters (match Chapter 4)
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

    diag_sw, df_sw = run_swingup(plant, logger)
    diag_lqr, df_lqr = run_lqr(plant, logger)

    summary_df = pd.DataFrame(
        [
            {
                "scenario": "swingup",
                "runtime_s": diag_sw["runtime_s"],
                "n_points": diag_sw["n_points"],
                "rtol": diag_sw["rtol"],
                "atol": diag_sw["atol"],
            },
            {
                "scenario": "lqr",
                "runtime_s": diag_lqr["runtime_s"],
                "n_points": diag_lqr["n_points"],
                "rtol": diag_lqr["rtol"],
                "atol": diag_lqr["atol"],
                "settling_time_theta_eps0.05": diag_lqr["settling_time_theta_eps0.05"],
                "x_max_abs": diag_lqr["x_max_abs"],
            },
        ]
    )
    summary_df.to_csv("data/ch6_inverted_summary.csv", index=False)


if __name__ == "__main__":
    main()
