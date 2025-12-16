# scripts/ch6_06_double_pendulum_chaos.py
#
# Double pendulum: sensitivity to initial conditions + energy drift.
# Outputs:
#   data/ch6_double_chaos_summary.csv
#   data/ch6_double_chaos_timeseries.csv
#   figs/ch6_double_angle_difference.png
#   figs/ch6_double_energy_drift.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dss.models import DoublePendulum
from dss.core.experiments import run_simulation_with_diagnostics
from dss.core.logger import SimulationLogger


def main() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger = SimulationLogger(log_dir="logs")

    # Parameters (match Chapter 4)
    dp = DoublePendulum(
        length1=1.0,
        mass1=1.0,
        length2=1.0,
        mass2=1.0,
        mode="ideal",
        damping1=0.0,
        damping2=0.0,
        coulomb1=0.0,
        coulomb2=0.0,
        gravity=9.81,
    )

    # Two very close initial conditions
    base_x0 = np.array([0.7, 0.0, 0.7, 0.0], dtype=float)
    delta = 1e-3
    x0_a = base_x0.copy()
    x0_b = base_x0.copy()
    x0_b[0] += delta  # small perturbation in theta1

    T_total = 20.0
    fps = 400
    rtol = 1e-6
    atol = 1e-8

    sol_a, diag_a = run_simulation_with_diagnostics(
        system=dp,
        initial_state=x0_a,
        T=T_total,
        fps=fps,
        rtol=rtol,
        atol=atol,
        logger=logger,
        experiment_name="double_pendulum_chaos_A",
        extra_meta={"chapter": 6, "kind": "double_chaos", "branch": "A"},
    )
    sol_b, diag_b = run_simulation_with_diagnostics(
        system=dp,
        initial_state=x0_b,
        T=T_total,
        fps=fps,
        rtol=rtol,
        atol=atol,
        logger=logger,
        experiment_name="double_pendulum_chaos_B",
        extra_meta={"chapter": 6, "kind": "double_chaos", "branch": "B"},
    )

    t = sol_a.t
    assert np.allclose(t, sol_b.t)

    Y_a = sol_a.y  # shape (4, N)
    Y_b = sol_b.y

    # Difference norm over time
    diff = Y_b - Y_a
    diff_norm = np.linalg.norm(diff, axis=0)

    # Energy drift for one trajectory
    energies = []
    for k in range(Y_a.shape[1]):
        energies.append(dp.energy_check(Y_a[:, k]))
    energies = np.asarray(energies, dtype=float)  # (N, 3) if [T, V, E]
    E = energies[:, -1]
    energy_drift = E - E[0]
    max_energy_error = float(np.max(np.abs(energy_drift)))

    # Save summary + timeseries
    summary_df = pd.DataFrame(
        [
            {
                "delta_theta1_init": delta,
                "runtime_A": diag_a["runtime_s"],
                "runtime_B": diag_b["runtime_s"],
                "n_points": diag_a["n_points"],
                "rtol": diag_a["rtol"],
                "atol": diag_a["atol"],
                "max_energy_error": max_energy_error,
            }
        ]
    )
    summary_df.to_csv("data/ch6_double_chaos_summary.csv", index=False)

    ts_df = pd.DataFrame(
        {
            "t": t,
            # Branch A: full state (for qualitative plots and energy)
            "theta1_A": Y_a[0, :],
            "theta1_dot_A": Y_a[1, :],
            "theta2_A": Y_a[2, :],
            "theta2_dot_A": Y_a[3, :],
            # Branch B: full state (used for diff_norm, optional plots)
            "theta1_B": Y_b[0, :],
            "theta1_dot_B": Y_b[1, :],
            "theta2_B": Y_b[2, :],
            "theta2_dot_B": Y_b[3, :],
            # Diagnostics
            "diff_norm": diff_norm,
            "E": E,
            "E_minus_E0": energy_drift,
        }
    )
    ts_df.to_csv("data/ch6_double_chaos_timeseries.csv", index=False)

    ts_df.to_csv("data/ch6_double_chaos_timeseries.csv", index=False)

    # Plot: divergence of trajectories
    plt.figure()
    plt.semilogy(t, diff_norm)
    plt.xlabel("t [s]")
    plt.ylabel("||Δstate||")
    plt.title("Double pendulum: sensitivity to initial conditions")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figs/ch6_double_angle_difference.png", dpi=300)
    plt.close()

    # Plot: energy deviation
    plt.figure()
    plt.plot(t, energy_drift)
    plt.xlabel("t [s]")
    plt.ylabel("E(t) - E(0) [J]")
    plt.title("Double pendulum: energy deviation (ideal mode)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figs/ch6_double_energy_drift.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
