# scripts/ch6_01_pendulum_small_angle.py
#
# Single pendulum: small-angle functional + energy test for Chapter 6.
# Produces:
#   data/ch6_pendulum_small_angle_summary.csv
#   data/ch6_pendulum_small_angle_timeseries.csv
#   figs/ch6_pendulum_theta.png
#   figs/ch6_pendulum_energy_drift.png

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adjust this import to your project structure:
from dss.models.pendulum import Pendulum  # e.g. dss.systems.pendulum if needed

from dss.core.experiments import run_simulation_with_diagnostics
from dss.core.logger import SimulationLogger

from typing import Optional

def estimate_period_from_peaks(t: np.ndarray, theta: np.ndarray) -> Optional[float]:
    """
    Very simple period estimator based on local maxima of theta(t).

    Returns
    -------
    float or None
        Estimated period or None if not enough peaks are found.
    """
    t = np.asarray(t, dtype=float)
    theta = np.asarray(theta, dtype=float)

    if t.size < 3:
        return None

    peak_times: list[float] = []
    for k in range(1, len(theta) - 1):
        if theta[k] > theta[k - 1] and theta[k] > theta[k + 1]:
            peak_times.append(float(t[k]))

    if len(peak_times) < 2:
        return None

    diffs = np.diff(peak_times)
    if diffs.size == 0:
        return None

    return float(np.mean(diffs))


def main() -> None:
    # Output directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger = SimulationLogger(log_dir="logs")

    # ------------------------------------------------------------------
    # 1) Define system parameters (match Chapter 4 / GUI defaults)
    # ------------------------------------------------------------------
    L = 1.0   # [m]
    m = 1.0   # [kg]
    g = 9.81  # [m/s^2]

    pend = Pendulum(
        length=L,
        mass=m,
        mode="ideal",      # gravity only
        damping=0.0,
        coulomb=0.0,
        gravity=g,
    )

    # Small-angle initial condition (linear regime)
    theta0 = 0.1   # [rad]
    omega0 = 0.0   # [rad/s]
    x0 = np.array([theta0, omega0], dtype=float)

    # Simulation settings
    T_total = 10.0  # [s]
    fps = 200       # samples per second
    rtol = 1e-6
    atol = 1e-8

    # ------------------------------------------------------------------
    # 2) Run simulation with diagnostics + logging
    # ------------------------------------------------------------------
    sol, diag = run_simulation_with_diagnostics(
        system=pend,
        initial_state=x0,
        T=T_total,
        fps=fps,
        rtol=rtol,
        atol=atol,
        logger=logger,
        experiment_name="pendulum_small_angle",
        extra_meta={"chapter": 6, "kind": "functional_test"},
    )

    t = sol.t                      # shape (N,)
    theta = sol.y[0, :]            # theta(t)
    theta_dot = sol.y[1, :]        # theta_dot(t)

    # ------------------------------------------------------------------
    # 3) Period estimate and analytic reference
    # ------------------------------------------------------------------
    T_sim = estimate_period_from_peaks(t, theta)
    T_th = 2.0 * np.pi * np.sqrt(L / g)  # small-angle formula

    if T_sim is not None:
        rel_period_error = abs(T_sim - T_th) / T_th
    else:
        rel_period_error = np.nan

    # ------------------------------------------------------------------
    # 4) Energies and energy drift
    # ------------------------------------------------------------------
    # Assuming: pendulum.energy_check(state) -> [T, V, E]
    energies = []
    for k in range(theta.size):
        state_k = sol.y[:, k]
        energies.append(pend.energy_check(state_k))
    energies = np.asarray(energies, dtype=float)  # shape (N, 3) if [T, V, E]

    KE = energies[:, 0]
    PE = energies[:, 1]
    E = energies[:, 2]
    energy_drift = E - E[0]
    max_energy_error = float(np.max(np.abs(energy_drift)))

    # ------------------------------------------------------------------
    # 5) Save summary CSV (for table in Chapter 6)
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(
        [
            {
                "L": L,
                "m": m,
                "g": g,
                "theta0": theta0,
                "omega0": omega0,
                "T_sim": T_sim,
                "T_th": T_th,
                "rel_period_error": rel_period_error,
                "max_energy_error": max_energy_error,
                "runtime_s": diag["runtime_s"],
                "n_points": diag["n_points"],
                "rtol": diag["rtol"],
                "atol": diag["atol"],
            }
        ]
    )
    summary_df.to_csv(
        "data/ch6_pendulum_small_angle_summary.csv", index=False
    )

    # Optional: full time series for further analysis
    timeseries_df = pd.DataFrame(
        {
            "t": t,
            "theta": theta,
            "theta_dot": theta_dot,
            "KE": KE,
            "PE": PE,
            "E": E,
            "E_minus_E0": energy_drift,
        }
    )
    timeseries_df.to_csv(
        "data/ch6_pendulum_small_angle_timeseries.csv", index=False
    )

    # ------------------------------------------------------------------
    # 6) Plots for Chapter 6 figures
    # ------------------------------------------------------------------
    # theta(t)
    plt.figure()
    plt.plot(t, theta)
    plt.xlabel("t [s]")
    plt.ylabel("theta [rad]")
    plt.title("Single pendulum: small-angle oscillation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figs/ch6_pendulum_theta.png", dpi=300)
    plt.close()

    # Energy deviation E(t) - E(0)
    plt.figure()
    plt.plot(t, energy_drift)
    plt.xlabel("t [s]")
    plt.ylabel("E(t) - E(0) [J]")
    plt.title("Single pendulum: energy deviation (ideal mode)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figs/ch6_pendulum_energy_drift.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
