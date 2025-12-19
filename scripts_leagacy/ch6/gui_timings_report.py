# scripts/ch6/gui_timings_report.py
#
# Chapter 6 (performance appendix): GUI / plotting timing micro-benchmarks.
#
# Goal:
# - quantify the cost of generating and serializing typical Matplotlib figures
#   used by the Streamlit dashboards (a proxy for "how fast Streamlit can load").
#
# Produces (under artifacts/ch6):
#   data/ch6_gui_timings.csv
#   figs/ch6_gui_timings_breakdown.png
#
# Important:
# - This is NOT a full end-to-end Streamlit benchmark. Streamlit rendering time
#   depends on browser, network, websocket batching, etc.
# - What we measure here are the deterministic Python-side costs:
#     (1) simulation runtime (solve_ivp),
#     (2) figure creation + plotting,
#     (3) PNG serialization (bytes sent to the frontend).
#
# Run:
#   python scripts/ch6/gui_timings_report.py

from __future__ import annotations

import sys
from pathlib import Path

# Make repository root importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import io
from dataclasses import dataclass
from time import perf_counter
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
class Case:
    key: str
    title: str
    build: Callable[[], Tuple[object, np.ndarray]]
    T: float
    fps: int


def _cases() -> List[Case]:
    def pend() -> Tuple[object, np.ndarray]:
        sys = Pendulum(length=1.0, mass=1.0, mode="ideal", damping=0.0, coulomb=0.0, gravity=9.81)
        return sys, np.array([0.5, 0.0], dtype=float)

    def vdp() -> Tuple[object, np.ndarray]:
        sys = VanDerPol(L=1.0, C=1.0, mu=1.0)
        return sys, np.array([1.0, 0.0], dtype=float)

    def motor() -> Tuple[object, np.ndarray]:
        sys = DCMotor(R=1.0, L=0.5, Ke=0.1, Kt=0.1, Im=0.01, bm=0.001, voltage_func=6.0)
        return sys, np.array([0.0, 0.0], dtype=float)

    def inv() -> Tuple[object, np.ndarray]:
        sys = InvertedPendulum(mode="damped_both", length=0.3, mass=0.2, cart_mass=0.5, gravity=9.81, b_cart=0.0, b_pend=0.0)
        return sys, np.array([0.0, 0.0, 0.2, 0.0], dtype=float)

    def dbl() -> Tuple[object, np.ndarray]:
        sys = DoublePendulum(length1=1.0, mass1=1.0, length2=1.0, mass2=1.0, mode="ideal", damping1=0.0, damping2=0.0, gravity=9.81)
        return sys, np.array([0.7, 0.0, 0.7, 0.0], dtype=float)

    def lor() -> Tuple[object, np.ndarray]:
        sys = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        return sys, np.array([1.0, 1.0, 1.0], dtype=float)

    return [
        Case("pendulum", "Pendulum", pend, T=10.0, fps=200),
        Case("vdp", "Van der Pol", vdp, T=10.0, fps=200),
        Case("dc_motor", "DC motor", motor, T=2.0, fps=500),
        Case("inverted", "Inverted pendulum (open-loop)", inv, T=10.0, fps=200),
        Case("double_pendulum", "Double pendulum", dbl, T=10.0, fps=200),
        Case("lorenz", "Lorenz", lor, T=5.0, fps=400),
    ]


def _serialize_png(fig: plt.Figure, *, dpi: int) -> Tuple[float, int]:
    buf = io.BytesIO()
    t0 = perf_counter()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    dt = perf_counter() - t0
    nbytes = buf.tell()
    buf.close()
    return float(dt), int(nbytes)


def _make_preview_figure(case: Case, t: np.ndarray, y: np.ndarray) -> plt.Figure:
    """Create a compact 'dashboard-like' preview figure."""
    fig, (axL, axR) = plt.subplots(nrows=1, ncols=2, figsize=(9.0, 3.2))

    # Left: geometry/phase portrait proxy
    sys, _ = case.build()
    if hasattr(sys, "positions"):
        # plot the tip trajectory (when available)
        pts = []
        for k in range(y.shape[1]):
            ps = sys.positions(y[:, k])
            pts.append(ps[-1])
        pts = np.asarray(pts, dtype=float)
        axL.plot(pts[:, 0], pts[:, 1])
        axL.set_aspect("equal", adjustable="box")
        axL.set_title("Geometry / trajectory")
        axL.set_xlabel("x")
        axL.set_ylabel("y")
    else:
        axL.plot(y[0, :], y[1, :] if y.shape[0] > 1 else np.zeros_like(y[0, :]))
        axL.set_title("Phase portrait proxy")
        axL.set_xlabel("state[0]")
        axL.set_ylabel("state[1]")

    # Right: first state vs time
    axR.plot(t, y[0, :])
    axR.set_title("State[0] vs time")
    axR.set_xlabel("t [s]")
    axR.set_ylabel("state[0]")

    fig.suptitle(case.title)
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="RK45")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--repeat", type=int, default=10, help="Repeat PNG serialization this many times to estimate per-frame cost.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    paths = resolve_paths()
    paths.artifacts_data.mkdir(parents=True, exist_ok=True)
    paths.artifacts_figs.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    for case in _cases():
        sys, x0 = case.build()

        # 1) simulation
        sol, diag = run_simulation_with_diagnostics(
            system=sys,
            initial_state=x0,
            T=case.T,
            fps=case.fps,
            method=args.method,
            rtol=args.rtol,
            atol=args.atol,
        )
        t = sol.t
        y = sol.y

        # 2) figure build time
        t0 = perf_counter()
        fig = _make_preview_figure(case, t, y)
        build_s = perf_counter() - t0

        # 3) serialization time (single)
        save_s, nbytes = _serialize_png(fig, dpi=args.dpi)

        # 4) repeated serialization (proxy for animation / repeated reruns)
        rep_times: List[float] = []
        for _ in range(int(args.repeat)):
            dt, _ = _serialize_png(fig, dpi=args.dpi)
            rep_times.append(dt)
        rep_mean = float(np.mean(rep_times)) if rep_times else float("nan")
        rep_p95 = float(np.percentile(rep_times, 95)) if rep_times else float("nan")

        plt.close(fig)

        rows.append(
            {
                "model": case.key,
                "title": case.title,
                "method": args.method,
                "rtol": float(args.rtol),
                "atol": float(args.atol),
                "T": float(case.T),
                "fps": int(case.fps),
                "n_points": int(t.size),
                "sim_runtime_s": float(diag["runtime_s"]),
                "plot_build_s": float(build_s),
                "png_save_s": float(save_s),
                "png_bytes": int(nbytes),
                "png_save_mean_s": float(rep_mean),
                "png_save_p95_s": float(rep_p95),
                "dpi": int(args.dpi),
                "repeat": int(args.repeat),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = paths.artifacts_data / "ch6_gui_timings.csv"
    df.to_csv(out_csv, index=False)

    # Plot: stacked-ish bars (simulation + plot + single PNG save)
    labels = df["model"].tolist()
    sim = df["sim_runtime_s"].to_numpy(dtype=float)
    build = df["plot_build_s"].to_numpy(dtype=float)
    save = df["png_save_s"].to_numpy(dtype=float)

    x = np.arange(len(labels))
    fig = plt.figure(figsize=(10.5, 3.8))
    ax = fig.add_subplot(111)
    ax.bar(x, sim, label="simulation")
    ax.bar(x, build, bottom=sim, label="plot build")
    ax.bar(x, save, bottom=sim + build, label="PNG serialize")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("time [s]")
    ax.set_title("Python-side timing breakdown (proxy for Streamlit plot load cost)")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    out_png = paths.artifacts_figs / "ch6_gui_timings_breakdown.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_png}")


if __name__ == "__main__":
    main()
