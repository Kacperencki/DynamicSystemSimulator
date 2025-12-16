# scripts/ch6_03_dc_motor_step.py
#
# DC motor: step response test (constant voltage).
# Outputs:
#   data/ch6_dc_motor_step_summary.csv
#   data/ch6_dc_motor_step_timeseries.csv
#   figs/ch6_dc_motor_step_response.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dss.models import DCMotor
from dss.core.experiments import run_simulation_with_diagnostics
from dss.core.logger import SimulationLogger

from typing import Optional
def estimate_time_constant(t: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Estimate first-order time constant as time to reach ~63% of steady state.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.size < 3:
        return None

    y_ss = float(np.mean(y[int(0.9 * len(y)) :]))  # average of last 10%
    target = 0.632 * y_ss

    # first index where y >= target
    idx = np.where(y >= target)[0]
    if idx.size == 0:
        return None
    return float(t[idx[0]])


def main() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger = SimulationLogger(log_dir="logs")

    # Motor parameters (example values; match with Chapter 4)
    R = 1.0
    L = 0.5
    Ke = 0.1
    Kt = 0.1
    Im = 0.01
    bm = 0.001

    V_step = 6.0  # [V]
    motor = DCMotor(
        R=R,
        L=L,
        Ke=Ke,
        Kt=Kt,
        Im=Im,
        bm=bm,
        voltage_func=V_step,
        load_func=0.0,
    )

    x0 = np.array([0.0, 0.0], dtype=float)  # [i, omega]
    T_total = 2.0
    fps = 500
    rtol = 1e-6
    atol = 1e-8

    sol, diag = run_simulation_with_diagnostics(
        system=motor,
        initial_state=x0,
        T=T_total,
        fps=fps,
        rtol=rtol,
        atol=atol,
        logger=logger,
        experiment_name="dc_motor_step",
        extra_meta={"chapter": 6, "kind": "dc_motor_step"},
    )

    t = sol.t
    i = sol.y[0, :]
    omega = sol.y[1, :]

    tau_est = estimate_time_constant(t, omega)

    summary_df = pd.DataFrame(
        [
            {
                "R": R,
                "L": L,
                "Ke": Ke,
                "Kt": Kt,
                "Im": Im,
                "bm": bm,
                "V_step": V_step,
                "omega_ss": float(np.mean(omega[int(0.9 * len(omega)) :])),
                "tau_est": tau_est,
                "runtime_s": diag["runtime_s"],
                "n_points": diag["n_points"],
                "rtol": diag["rtol"],
                "atol": diag["atol"],
            }
        ]
    )
    summary_df.to_csv("data/ch6_dc_motor_step_summary.csv", index=False)

    ts_df = pd.DataFrame({"t": t, "i": i, "omega": omega})
    ts_df.to_csv("data/ch6_dc_motor_step_timeseries.csv", index=False)

    # Plot step response i(t), omega(t)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax1.plot(t, i)
    ax1.set_ylabel("i [A]")
    ax1.grid(True)
    ax1.set_title("DC motor: step response")

    ax2.plot(t, omega)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("ω [rad/s]")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig("figs/ch6_dc_motor_step_response.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
