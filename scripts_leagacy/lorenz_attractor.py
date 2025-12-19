# scripts/ch6_07_lorenz_attractor.py
#
# Lorenz system: attractor and sensitivity to initial conditions.
# Outputs:
#   data/ch6_lorenz_summary.csv
#   data/ch6_lorenz_timeseries.csv
#   figs/ch6_lorenz_attractor_3d.png
#   figs/ch6_lorenz_difference.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

from dss.models.lorenz import Lorenz
from dss.core.experiments import run_simulation_with_diagnostics
from dss.core.logger import SimulationLogger


def main() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger = SimulationLogger(log_dir="logs")

    system = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)

    x0_a = np.array([1.0, 0.0, 0.0], dtype=float)
    x0_b = x0_a + np.array([1e-6, 0.0, 0.0], dtype=float)  # tiny perturbation

    T_total = 40.0
    fps = 400
    rtol = 1e-6
    atol = 1e-8

    sol_a, diag_a = run_simulation_with_diagnostics(
        system=system,
        initial_state=x0_a,
        T=T_total,
        fps=fps,
        rtol=rtol,
        atol=atol,
        logger=logger,
        experiment_name="lorenz_A",
        extra_meta={"chapter": 6, "kind": "lorenz", "branch": "A"},
    )
    sol_b, diag_b = run_simulation_with_diagnostics(
        system=system,
        initial_state=x0_b,
        T=T_total,
        fps=fps,
        rtol=rtol,
        atol=atol,
        logger=logger,
        experiment_name="lorenz_B",
        extra_meta={"chapter": 6, "kind": "lorenz", "branch": "B"},
    )

    t = sol_a.t
    assert np.allclose(t, sol_b.t)

    X_a = sol_a.y  # (3, N)
    X_b = sol_b.y
    diff_norm = np.linalg.norm(X_b - X_a, axis=0)

    # Save timeseries (for branch A, plus difference)
    df = pd.DataFrame(
        {
            "t": t,
            "x": X_a[0, :],
            "y": X_a[1, :],
            "z": X_a[2, :],
            "diff_norm": diff_norm,
        }
    )
    df.to_csv("data/ch6_lorenz_timeseries.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "sigma": system.sigma,
                "rho": system.rho,
                "beta": system.beta,
                "delta_x0": float(x0_b[0] - x0_a[0]),
                "runtime_A": diag_a["runtime_s"],
                "runtime_B": diag_b["runtime_s"],
                "n_points": diag_a["n_points"],
                "rtol": diag_a["rtol"],
                "atol": diag_a["atol"],
            }
        ]
    )
    summary_df.to_csv("data/ch6_lorenz_summary.csv", index=False)

    # 3D attractor plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X_a[0, :], X_a[1, :], X_a[2, :])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Lorenz attractor")
    plt.tight_layout()
    fig.savefig("figs/ch6_lorenz_attractor_3d.png", dpi=300)
    plt.close(fig)

    # Difference norm plot (sensitivity)
    plt.figure()
    plt.semilogy(t, diff_norm)
    plt.xlabel("t [s]")
    plt.ylabel("||Δstate||")
    plt.title("Lorenz system: sensitivity to initial conditions")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figs/ch6_lorenz_difference.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
