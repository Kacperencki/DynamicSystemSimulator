# scripts/ch6_05_performance.py
#
# Performance benchmarks for Chapter 6:
#   - Runtime per model (pendulum, double pendulum, Van der Pol,
#     DC motor, inverted pendulum)
#   - Runtime vs tolerance for the single pendulum
#
# Outputs:
#   data/ch6_performance_models.csv
#   data/ch6_performance_pendulum_tol.csv
#   figs/ch6_runtime_per_model.png
#   figs/ch6_runtime_vs_tol_pendulum.png

import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dss.models.pendulum import Pendulum          # :contentReference[oaicite:1]{index=1}
from dss.models.double_pendulum import DoublePendulum  # :contentReference[oaicite:2]{index=2}
from dss.models.vanderpol_circuit import VanDerPol    # :contentReference[oaicite:3]{index=3}
from dss.models.dc_motor import DCMotor                # :contentReference[oaicite:4]{index=4}
from dss.models.inverted_pendulum import InvertedPendulum  # :contentReference[oaicite:5]{index=5}

from dss.core.experiments import run_simulation_with_diagnostics
from dss.core.logger import SimulationLogger


# ---------------------------------------------------------------------------
# Helpers to construct default models for benchmarking
# ---------------------------------------------------------------------------

def make_default_models() -> List[Dict[str, Any]]:
    """
    Create a list of benchmark scenarios.
    Each entry defines:
      - name        : label for tables/plots
      - system      : model instance
      - x0          : initial state
      - T           : total simulation time [s]
      - fps         : sampling frequency [Hz]
    Parameters are chosen to be simple and representative, not perfect physics.
    """
    models: List[Dict[str, Any]] = []

    # Single pendulum (ideal)
    L = 1.0
    m = 1.0
    g = 9.81
    pend = Pendulum(
        length=L,
        mass=m,
        mode="ideal",
        damping=0.0,
        coulomb=0.0,
        gravity=g,
    )
    models.append(
        dict(
            name="pendulum_ideal",
            system=pend,
            x0=np.array([0.5, 0.0]),  # [theta, theta_dot]
            T=10.0,
            fps=200,
        )
    )

    # Double pendulum (ideal)
    dp = DoublePendulum(
        length1=1.0,
        mass1=1.0,
        length2=1.0,
        mass2=1.0,
        mode="ideal",
        gravity=9.81,
    )
    models.append(
        dict(
            name="double_pendulum_ideal",
            system=dp,
            x0=np.array([0.5, 0.0, 1.0, 0.0]),  # [theta1, theta1_dot, theta2, theta2_dot]
            T=10.0,
            fps=300,
        )
    )

    # Van der Pol oscillator
    vdp = VanDerPol(L=1.0, C=1.0, mu=1.0)
    models.append(
        dict(
            name="vanderpol",
            system=vdp,
            x0=np.array([1.0, 0.0]),  # [v, iL]
            T=40.0,
            fps=200,
        )
    )

    # DC motor with constant voltage
    motor = DCMotor(
        R=1.0,
        L=0.5,
        Ke=0.1,
        Kt=0.1,
        Im=0.01,      # rotor inertia
        bm=0.001,
        voltage_func=6.0,
        load_func=0.0,
    )
    models.append(
        dict(
            name="dc_motor",
            system=motor,
            x0=np.array([0.0, 0.0]),  # [i, omega]
            T=2.0,
            fps=500,
        )
    )

    # Inverted pendulum (damped cart+pendulum, no external drive)
    ip = InvertedPendulum(
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
    models.append(
        dict(
            name="inverted_pendulum_damped",
            system=ip,
            x0=np.array([0.0, 0.0, 0.2, 0.0]),  # [x, x_dot, theta (rad from UP), theta_dot]
            T=10.0,
            fps=300,
        )
    )

    return models


# ---------------------------------------------------------------------------
# Benchmark 1: runtime per model at a common tolerance
# ---------------------------------------------------------------------------

def benchmark_runtime_per_model(
    logger: SimulationLogger,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> pd.DataFrame:
    models = make_default_models()
    rows = []

    for cfg in models:
        name = cfg["name"]
        system = cfg["system"]
        x0 = cfg["x0"]
        T = float(cfg["T"])
        fps = int(cfg["fps"])

        sol, diag = run_simulation_with_diagnostics(
            system=system,
            initial_state=x0,
            T=T,
            fps=fps,
            rtol=rtol,
            atol=atol,
            logger=logger,
            experiment_name=f"perf_{name}",
            extra_meta={"chapter": 6, "kind": "performance_models"},
        )

        row = {
            "model": name,
            "state_dim": int(sol.y.shape[0]),
            "T": T,
            "fps": fps,
            "rtol": rtol,
            "atol": atol,
            "runtime_s": diag["runtime_s"],
            "n_points": diag["n_points"],
            "t_start": diag["t_start"],
            "t_end": diag["t_end"],
            "max_energy_error": diag.get("max_energy_error", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("data/ch6_performance_models.csv", index=False)
    return df


def plot_runtime_per_model(df: pd.DataFrame) -> None:
    plt.figure()
    x = np.arange(len(df))
    plt.bar(x, df["runtime_s"])
    plt.xticks(x, df["model"], rotation=30, ha="right")
    plt.ylabel("runtime [s]")
    plt.title("Runtime per model (common tolerances)")
    plt.tight_layout()
    plt.savefig("figs/ch6_runtime_per_model.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Benchmark 2: runtime vs tolerance for the single pendulum
# ---------------------------------------------------------------------------

def benchmark_pendulum_tolerance(
    logger: SimulationLogger,
    tolerances=(1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
) -> pd.DataFrame:
    L = 1.0
    m = 1.0
    g = 9.81
    pend = Pendulum(
        length=L,
        mass=m,
        mode="ideal",
        damping=0.0,
        coulomb=0.0,
        gravity=g,
    )
    x0 = np.array([0.5, 0.0])
    T_total = 10.0
    fps = 200

    rows = []

    for rtol in tolerances:
        sol, diag = run_simulation_with_diagnostics(
            system=pend,
            initial_state=x0,
            T=T_total,
            fps=fps,
            rtol=float(rtol),
            atol=float(rtol) * 1e-2,  # simple scaling
            logger=logger,
            experiment_name="perf_pendulum_tol",
            extra_meta={
                "chapter": 6,
                "kind": "performance_pendulum_tol",
                "rtol_sweep": True,
            },
        )

        row = {
            "rtol": float(rtol),
            "atol": float(rtol) * 1e-2,
            "runtime_s": diag["runtime_s"],
            "n_points": diag["n_points"],
            "t_start": diag["t_start"],
            "t_end": diag["t_end"],
            "max_energy_error": diag.get("max_energy_error", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("rtol", ascending=False)
    df.to_csv("data/ch6_performance_pendulum_tol.csv", index=False)
    return df


def plot_pendulum_runtime_vs_tol(df: pd.DataFrame) -> None:
    plt.figure()
    # log scale on x (rtol), optionally on y
    plt.loglog(df["rtol"], df["runtime_s"], marker="o")
    plt.xlabel("rtol")
    plt.ylabel("runtime [s]")
    plt.title("Single pendulum: runtime vs tolerance")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figs/ch6_runtime_vs_tol_pendulum.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger = SimulationLogger(log_dir="logs")

    # Benchmark 1: per-model runtime at common tolerances
    df_models = benchmark_runtime_per_model(logger, rtol=1e-4, atol=1e-6)
    plot_runtime_per_model(df_models)

    # Benchmark 2: pendulum runtime vs tolerance
    df_tol = benchmark_pendulum_tolerance(logger)
    plot_pendulum_runtime_vs_tol(df_tol)


if __name__ == "__main__":
    main()
