# scripts/ch6/convergence_sweeps.py
#
# Chapter 6 (6.3): Solver convergence sweeps.
#
# Produces (under artifacts/ch6):
#   data/ch6_convergence_summary.csv
#   figs/ch6_convergence_residuals.png
#
# The summary CSV contains per-model/per-method/per-rtol error metrics against a
# strict "reference" solution evaluated on the same time grid.
#
# Run:
#   python scripts/ch6/convergence_sweeps.py
#
# (Also runnable via: python scripts/ch6/run_all.py --key convergence_sweeps --collect)

from __future__ import annotations

import sys
from pathlib import Path

# Make repository root importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dss.core.experiments import run_simulation_with_diagnostics
from dss.models.pendulum import Pendulum
from dss.models.vanderpol_circuit import VanDerPol
from dss.models.dc_motor import DCMotor
from dss.models.inverted_pendulum import InvertedPendulum
from dss.models.double_pendulum import DoublePendulum
from dss.models.lorenz import Lorenz

from scripts.ch6.paths import resolve_paths


@dataclass(frozen=True)
class ModelCase:
    key: str
    title: str
    build: Callable[[], Tuple[object, np.ndarray]]
    T: float
    fps: int


def _parse_floats(csv: str) -> List[float]:
    out: List[float] = []
    for token in csv.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    return out


def _parse_strs(csv: str) -> List[str]:
    return [t.strip() for t in csv.split(",") if t.strip()]


def _error_norm(y: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """Return e(t) = ||y(t) - y_ref(t)||_2 for each time sample.

    Inputs are expected as (n_state, n_time).
    """
    diff = (y - y_ref).T  # (n_time, n_state)
    return np.linalg.norm(diff, axis=1)


def _case_pendulum() -> ModelCase:
    def build() -> Tuple[object, np.ndarray]:
        sys = Pendulum(length=1.0, mass=1.0, mode="ideal", damping=0.0, coulomb=0.0, gravity=9.81)
        x0 = np.array([0.5, 0.0], dtype=float)
        return sys, x0

    return ModelCase(key="pendulum", title="Pendulum", build=build, T=10.0, fps=200)


def _case_vdp() -> ModelCase:
    def build() -> Tuple[object, np.ndarray]:
        sys = VanDerPol(L=1.0, C=1.0, mu=1.0)
        x0 = np.array([1.0, 0.0], dtype=float)
        return sys, x0

    return ModelCase(key="vdp", title="Van der Pol", build=build, T=10.0, fps=200)


def _case_dc_motor() -> ModelCase:
    def build() -> Tuple[object, np.ndarray]:
        sys = DCMotor(R=1.0, L=0.5, Ke=0.1, Kt=0.1, Im=0.01, bm=0.001, voltage_func=6.0)
        x0 = np.array([0.0, 0.0], dtype=float)  # [i, omega]
        return sys, x0

    return ModelCase(key="dc_motor", title="DC motor", build=build, T=2.0, fps=500)


def _case_inverted_open_loop() -> ModelCase:
    def build() -> Tuple[object, np.ndarray]:
        sys = InvertedPendulum(
            mode="damped_both",
            length=0.3,
            mass=0.2,
            cart_mass=0.5,
            gravity=9.81,
            b_cart=0.0,
            b_pend=0.0,
            coulomb_cart=0.0,
            coulomb_pend=0.0,
        )
        x0 = np.array([0.0, 0.0, 0.2, 0.0], dtype=float)  # [x, xdot, theta(up), thetadot]
        return sys, x0

    return ModelCase(key="inverted", title="Inverted pendulum (open-loop)", build=build, T=10.0, fps=200)


def _case_double_pendulum() -> ModelCase:
    def build() -> Tuple[object, np.ndarray]:
        sys = DoublePendulum(
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
        x0 = np.array([0.7, 0.0, 0.7, 0.0], dtype=float)
        return sys, x0

    return ModelCase(key="double_pendulum", title="Double pendulum", build=build, T=10.0, fps=200)


def _case_lorenz() -> ModelCase:
    def build() -> Tuple[object, np.ndarray]:
        sys = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        x0 = np.array([1.0, 1.0, 1.0], dtype=float)
        return sys, x0

    return ModelCase(key="lorenz", title="Lorenz", build=build, T=5.0, fps=400)


ALL_CASES: Dict[str, ModelCase] = {
    c.key: c
    for c in [
        _case_pendulum(),
        _case_vdp(),
        _case_dc_motor(),
        _case_inverted_open_loop(),
        _case_double_pendulum(),
        _case_lorenz(),
    ]
}


def _run(case: ModelCase, *, method: str, rtol: float, atol: float) -> Tuple[np.ndarray, np.ndarray, float]:
    system, x0 = case.build()
    sol, diag = run_simulation_with_diagnostics(
        system=system,
        initial_state=x0,
        T=case.T,
        fps=case.fps,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return sol.t, sol.y, float(diag["runtime_s"])


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        default="pendulum,vdp,dc_motor,inverted,double_pendulum,lorenz",
        help="Comma-separated model keys. Options: %s" % ", ".join(sorted(ALL_CASES.keys())),
    )
    p.add_argument("--methods", default="RK45,DOP853,Radau", help="Comma-separated solve_ivp methods to compare.")
    p.add_argument("--rtol-grid", default="1e-2,1e-4,1e-6,1e-8", help="Comma-separated rtol values.")
    p.add_argument("--atol", type=float, default=1e-10, help="Absolute tolerance for sweep runs (kept fixed).")
    p.add_argument("--ref-method", default="DOP853", help="Method used for the strict reference solution.")
    p.add_argument("--ref-rtol", type=float, default=1e-12, help="Reference rtol.")
    p.add_argument("--ref-atol", type=float, default=1e-14, help="Reference atol.")
    p.add_argument("--plot-models", default="pendulum,lorenz", help="Comma-separated subset to include in the residual plot.")
    args = p.parse_args(list(argv) if argv is not None else None)

    paths = resolve_paths()
    paths.artifacts_data.mkdir(parents=True, exist_ok=True)
    paths.artifacts_figs.mkdir(parents=True, exist_ok=True)
    paths.artifacts_logs.mkdir(parents=True, exist_ok=True)

    model_keys = _parse_strs(args.models)
    methods = _parse_strs(args.methods)
    rtol_grid = _parse_floats(args.rtol_grid)
    plot_models = _parse_strs(args.plot_models)

    rows: List[Dict[str, object]] = []
    plot_series: Dict[Tuple[str, str, float], Tuple[np.ndarray, np.ndarray]] = {}
    references: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    # Reference solutions
    for mk in model_keys:
        case = ALL_CASES[mk]
        t_ref, y_ref, rt_ref = _run(case, method=args.ref_method, rtol=args.ref_rtol, atol=args.ref_atol)
        references[mk] = (t_ref, y_ref, rt_ref)

    # Sweep solutions
    for mk in model_keys:
        case = ALL_CASES[mk]
        t_ref, y_ref, rt_ref = references[mk]

        for method in methods:
            for rtol in rtol_grid:
                t, y, rt = _run(case, method=method, rtol=float(rtol), atol=float(args.atol))

                if t.shape != t_ref.shape or np.max(np.abs(t - t_ref)) > 1e-12:
                    raise RuntimeError(f"Time grid mismatch for {mk} ({method}, rtol={rtol}).")

                e = _error_norm(y, y_ref)

                if mk in plot_models and method in {"RK45", "DOP853", "Radau"}:
                    plot_series[(mk, method, float(rtol))] = (t, e)

                rows.append(
                    {
                        "model": mk,
                        "model_title": case.title,
                        "method": method,
                        "rtol": float(rtol),
                        "atol": float(args.atol),
                        "ref_method": args.ref_method,
                        "ref_rtol": float(args.ref_rtol),
                        "ref_atol": float(args.ref_atol),
                        "T": float(case.T),
                        "fps": int(case.fps),
                        "runtime_s": float(rt),
                        "runtime_ref_s": float(rt_ref),
                        "e_max": float(np.max(e)),
                        "e_rms": float(np.sqrt(np.mean(e ** 2))),
                    }
                )

    df = pd.DataFrame(rows)
    out_csv = paths.artifacts_data / "ch6_convergence_summary.csv"
    df.to_csv(out_csv, index=False)

    # Residual plot
    fig, axes = plt.subplots(nrows=len(plot_models), ncols=1, figsize=(9.2, 3.3 * max(1, len(plot_models))))
    if len(plot_models) == 1:
        axes = [axes]

    for ax, mk in zip(axes, plot_models):
        case = ALL_CASES[mk]
        for method in methods:
            if method not in {"RK45", "DOP853", "Radau"}:
                continue
            for rtol in rtol_grid:
                key = (mk, method, float(rtol))
                if key not in plot_series:
                    continue
                t, e = plot_series[key]
                ax.semilogy(t, e, label=f"{method}, rtol={rtol:g}")

        ax.set_title(f"{case.title}: residual e(t) vs reference")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("e(t) = ||Δstate||")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out_png = paths.artifacts_figs / "ch6_convergence_residuals.png"
    fig.savefig(out_png, dpi=250)
    plt.close(fig)

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_png}")


if __name__ == "__main__":
    main()
