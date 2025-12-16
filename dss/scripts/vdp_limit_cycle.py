# scripts/ch6_02_vdp_limit_cycle.py
#
# Van der Pol oscillator: convergence to limit cycle from different initial states.
# Outputs:
#   data/ch6_vdp_limit_cycle_summary.csv
#   data/ch6_vdp_limit_cycle_timeseries.csv
#   figs/ch6_vdp_phase_portrait.png
#   figs/ch6_vdp_time_series.png

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dss.models import VanDerPol
from dss.core.experiments import run_simulation_with_diagnostics
from dss.core.logger import SimulationLogger


def main() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger = SimulationLogger(log_dir="logs")

    # System parameters (match what you describe in Chapter 4)
    L = 1.0
    C = 1.0
    mu = 1.0
    vdp = VanDerPol(L=L, C=C, mu=mu)

    # Several initial conditions to show convergence to same limit cycle
    initial_conditions: List[Tuple[float, float]] = [
        (1.0, 0.0),
        (0.5, 0.5),
        (-1.0, 0.0),
    ]

    T_total = 40.0  # long enough to reach limit cycle
    fps = 300
    rtol = 1e-6
    atol = 1e-8

    summary_rows = []
    ts_rows = []

    for idx, (v0, iL0) in enumerate(initial_conditions):
        x0 = np.array([v0, iL0], dtype=float)

        sol, diag = run_simulation_with_diagnostics(
            system=vdp,
            initial_state=x0,
            T=T_total,
            fps=fps,
            rtol=rtol,
            atol=atol,
            logger=logger,
            experiment_name=f"vdp_limit_cycle_ic{idx}",
            extra_meta={"chapter": 6, "kind": "vdp_limit_cycle", "ic_index": idx},
        )

        t = sol.t
        v = sol.y[0, :]
        iL = sol.y[1, :]

        # Use last half of the trajectory to approximate limit cycle amplitude
        n = v.size
        tail = slice(n // 2, n)
        v_tail = v[tail]
        iL_tail = iL[tail]

        v_min, v_max = float(v_tail.min()), float(v_tail.max())
        iL_min, iL_max = float(iL_tail.min()), float(iL_tail.max())

        summary_rows.append(
            {
                "ic_index": idx,
                "v0": v0,
                "iL0": iL0,
                "v_min_tail": v_min,
                "v_max_tail": v_max,
                "iL_min_tail": iL_min,
                "iL_max_tail": iL_max,
                "runtime_s": diag["runtime_s"],
                "n_points": diag["n_points"],
                "rtol": diag["rtol"],
                "atol": diag["atol"],
            }
        )

        for tk, vk, iLk in zip(t, v, iL):
            ts_rows.append(
                {
                    "ic_index": idx,
                    "t": tk,
                    "v": vk,
                    "iL": iLk,
                }
            )

    # Save summary + timeseries
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("data/ch6_vdp_limit_cycle_summary.csv", index=False)

    ts_df = pd.DataFrame(ts_rows)
    ts_df.to_csv("data/ch6_vdp_limit_cycle_timeseries.csv", index=False)

    # Phase portrait (all initial conditions)
    plt.figure()
    for idx in sorted(summary_df["ic_index"].unique()):
        mask = ts_df["ic_index"] == idx
        plt.plot(ts_df.loc[mask, "v"], ts_df.loc[mask, "iL"], alpha=0.8, label=f"IC {idx}")
    plt.xlabel("v [V]")
    plt.ylabel("i_L [A]")
    plt.title("Van der Pol: convergence to limit cycle")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/ch6_vdp_phase_portrait.png", dpi=300)
    plt.close()

    # Single time series from first IC (for Chapter 6 figure)
    mask0 = ts_df["ic_index"] == 0
    plt.figure()
    plt.plot(ts_df.loc[mask0, "t"], ts_df.loc[mask0, "v"], label="v(t)")
    plt.plot(ts_df.loc[mask0, "t"], ts_df.loc[mask0, "iL"], label="i_L(t)")
    plt.xlabel("t [s]")
    plt.ylabel("state")
    plt.title("Van der Pol: time series (example IC)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/ch6_vdp_time_series.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
