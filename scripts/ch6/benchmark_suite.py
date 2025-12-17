# scripts/ch6/benchmarks_suite.py
"""Chapter 6.4 — Runtime benchmarks across models and solvers.

Runs a fixed-duration simulation for each model + solve_ivp method, on an explicit
evaluation grid, and reports wall-clock runtime and integrator counters.

Outputs (written to repo-root/artifacts/ch6/...):
  - artifacts/ch6/data/ch6_benchmarks_runtime.csv
  - artifacts/ch6/data/ch6_benchmarks_runtime_raw.csv
  - artifacts/ch6/figs/ch6_benchmarks_runtime.png
  - artifacts/ch6/figs/ch6_benchmarks_nfev.png
  - artifacts/ch6/logs/ch6_benchmarks_manifest.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dss.core.solver import Solver  # noqa: E402
from dss.models import (  # noqa: E402
    Pendulum,
    DoublePendulum,
    DCMotor,
    VanDerPol,
    Lorenz,
    InvertedPendulum,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    system: Any
    x0: np.ndarray
    T: float
    fps: int


def _build_models(*, T: float = 10.0) -> List[ModelSpec]:
    """Keep parameters aligned with behaviour_panels.py, but use a common horizon for benchmarking."""
    models: List[ModelSpec] = []

    pend = Pendulum(length=1.0, mass=1.0, gravity=9.81, mode="ideal")
    models.append(ModelSpec("pendulum", "Pendulum", pend, np.array([0.5, 0.0], float), T, 200))

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
    models.append(ModelSpec("double_pendulum", "Double pendulum", dp, np.array([1.2, 0.0, 1.0, 0.0], float), T, 200))

    inv = InvertedPendulum(
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
    models.append(ModelSpec("inverted_pendulum", "Inverted pendulum (plant)", inv, np.array([0.0, 0.0, np.pi - 0.2, 0.0], float), T, 200))

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
    # Use a lower fps for motor in long-horizon benchmarks to avoid excessive output arrays.
    models.append(ModelSpec("dc_motor", "DC motor", motor, np.array([0.0, 0.0], float), T, 500))

    vdp = VanDerPol(L=1.0, C=1.0, mu=2.0)
    models.append(ModelSpec("vanderpol", "Van der Pol", vdp, np.array([1.0, 0.0], float), T, 200))

    lor = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    models.append(ModelSpec("lorenz", "Lorenz", lor, np.array([1.0, 1.0, 1.0], float), T, 200))

    return models


def _time_grid(T: float, fps: int) -> np.ndarray:
    n = int(np.floor(T * fps)) + 1
    return np.linspace(0.0, float(T), n)


@dataclass(frozen=True)
class SolveStats:
    ok: bool
    runtime_s: float
    nfev: Optional[int]
    njev: Optional[int]
    nlu: Optional[int]
    message: str


def _one_run(system: Any, x0: np.ndarray, *, t_eval: np.ndarray, method: str, rtol: float, atol: float) -> SolveStats:
    t0 = perf_counter()
    try:
        sol = Solver(
            system=system,
            initial_conditions=x0,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            t_eval=t_eval,
            method=method,
            rtol=float(rtol),
            atol=float(atol),
        ).run()
        dt = perf_counter() - t0
        return SolveStats(
            ok=bool(getattr(sol, "success", True)),
            runtime_s=float(dt),
            nfev=int(getattr(sol, "nfev", 0)) if hasattr(sol, "nfev") else None,
            njev=int(getattr(sol, "njev", 0)) if hasattr(sol, "njev") else None,
            nlu=int(getattr(sol, "nlu", 0)) if hasattr(sol, "nlu") else None,
            message=str(getattr(sol, "message", "")),
        )
    except Exception as e:
        dt = perf_counter() - t0
        return SolveStats(False, float(dt), None, None, None, f"{type(e).__name__}: {e}")


def _ensure_dirs(root: Path) -> Dict[str, Path]:
    figs = root / "figs"
    data = root / "data"
    logs = root / "logs"
    figs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return {"figs": figs, "data": data, "logs": logs}


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--models", nargs="*", default=None, help="Subset of model keys to run (default: all)")
    p.add_argument("--methods", nargs="*", default=["RK45", "DOP853", "Radau", "BDF"], help="solve_ivp methods")
    p.add_argument("--T", type=float, default=10.0, help="simulation horizon [s]")
    p.add_argument("--rtol", type=float, default=1e-4, help="relative tolerance")
    p.add_argument("--atol", type=float, default=1e-6, help="absolute tolerance")
    p.add_argument("--repeats", type=int, default=3, help="repeats per (model, method); median is reported")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    artifacts_root = REPO_ROOT / "artifacts" / "ch6"
    out = _ensure_dirs(artifacts_root)

    models_all = _build_models(T=float(args.T))
    if args.models:
        wanted = set(args.models)
        models = [m for m in models_all if m.key in wanted]
    else:
        models = models_all

    methods = list(dict.fromkeys(args.methods))

    raw_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for m in models:
        t_eval = _time_grid(m.T, m.fps)

        for method in methods:
            stats_list: List[SolveStats] = []
            for r in range(int(args.repeats)):
                st = _one_run(m.system, m.x0, t_eval=t_eval, method=method, rtol=float(args.rtol), atol=float(args.atol))
                stats_list.append(st)
                raw_rows.append(
                    {
                        "model": m.key,
                        "method": method,
                        "repeat": r,
                        "T": m.T,
                        "fps": m.fps,
                        "rtol": float(args.rtol),
                        "atol": float(args.atol),
                        "ok": st.ok,
                        "runtime_s": st.runtime_s,
                        "nfev": st.nfev,
                        "njev": st.njev,
                        "nlu": st.nlu,
                        "message": st.message,
                    }
                )

            runtimes = np.array([s.runtime_s for s in stats_list], dtype=float)
            runtime_med = float(np.median(runtimes))
            ok_any = any(s.ok for s in stats_list)

            def _median_int(vals: List[Optional[int]]) -> Optional[int]:
                vv = [v for v in vals if v is not None]
                if not vv:
                    return None
                return int(np.median(np.array(vv, dtype=float)))

            summary_rows.append(
                {
                    "model": m.key,
                    "model_title": m.title,
                    "method": method,
                    "T": m.T,
                    "fps": m.fps,
                    "rtol": float(args.rtol),
                    "atol": float(args.atol),
                    "repeats": int(args.repeats),
                    "ok_any": ok_any,
                    "runtime_s_median": runtime_med,
                    "nfev_median": _median_int([s.nfev for s in stats_list]),
                    "njev_median": _median_int([s.njev for s in stats_list]),
                    "nlu_median": _median_int([s.nlu for s in stats_list]),
                }
            )

    df_raw = pd.DataFrame(raw_rows)
    df_sum = pd.DataFrame(summary_rows)

    df_raw.to_csv(out["data"] / "ch6_benchmarks_runtime_raw.csv", index=False)
    df_sum.to_csv(out["data"] / "ch6_benchmarks_runtime.csv", index=False)

    # Heatmaps (log10 scale for readability)
    model_keys = [m.key for m in models]
    method_keys = methods

    def _matrix(col: str) -> np.ndarray:
        mat = np.full((len(model_keys), len(method_keys)), np.nan, dtype=float)
        for i, mk in enumerate(model_keys):
            for j, meth in enumerate(method_keys):
                sub = df_sum[(df_sum["model"] == mk) & (df_sum["method"] == meth)]
                if len(sub) == 1:
                    v = float(sub.iloc[0][col])
                    mat[i, j] = v
        return mat

    runtime_mat = _matrix("runtime_s_median")
    nfev_mat = _matrix("nfev_median")

    # runtime heatmap
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    img = ax.imshow(np.log10(np.maximum(runtime_mat, 1e-12)), aspect="auto")
    ax.set_xticks(range(len(method_keys)), method_keys, rotation=30, ha="right")
    ax.set_yticks(range(len(model_keys)), model_keys)
    ax.set_title("Benchmark: log10(runtime [s])")
    ax.set_xlabel("Solver method")
    ax.set_ylabel("Model")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, out["figs"] / "ch6_benchmarks_runtime.png")

    # nfev heatmap (if available)
    fig2, ax2 = plt.subplots(figsize=(10.5, 3.8))
    mat2 = np.where(np.isfinite(nfev_mat), np.maximum(nfev_mat, 1.0), np.nan)
    img2 = ax2.imshow(np.log10(mat2), aspect="auto")
    ax2.set_xticks(range(len(method_keys)), method_keys, rotation=30, ha="right")
    ax2.set_yticks(range(len(model_keys)), model_keys)
    ax2.set_title("Benchmark: log10(nfev)")
    ax2.set_xlabel("Solver method")
    ax2.set_ylabel("Model")
    fig2.colorbar(img2, ax=ax2, fraction=0.046, pad=0.04)
    _save_fig(fig2, out["figs"] / "ch6_benchmarks_nfev.png")

    manifest = {
        "producer": "benchmarks_suite",
        "chapter": 6,
        "section": "6.4",
        "models": [
            {
                "key": m.key,
                "title": m.title,
                "system": {
                    "class": type(m.system).__name__,
                    "module": type(m.system).__module__,
                },
                "x0": [float(v) for v in np.asarray(m.x0).tolist()],
                "T": float(m.T),
                "fps": int(m.fps),
            }
            for m in models
        ],

        "methods": methods,
        "T": float(args.T),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "repeats": int(args.repeats),
        "outputs": {
            "summary_csv": "artifacts/ch6/data/ch6_benchmarks_runtime.csv",
            "raw_csv": "artifacts/ch6/data/ch6_benchmarks_runtime_raw.csv",
            "runtime_png": "artifacts/ch6/figs/ch6_benchmarks_runtime.png",
            "nfev_png": "artifacts/ch6/figs/ch6_benchmarks_nfev.png",
        },
    }
    (out["logs"] / "ch6_benchmarks_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
