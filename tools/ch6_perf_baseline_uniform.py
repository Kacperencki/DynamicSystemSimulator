#!/usr/bin/env python3
"""
Chapter 6.4 — Uniform baseline performance benchmark (runtime + nfev).

Runs each DSS model for the SAME horizon T and sampling fps (=> N = T*fps + 1).
Repeats each case and reports median runtime and median nfev when available.

Outputs (in --out directory):
- perf_baseline_uniform.csv
- table_perf_baseline_uniform.tex
- runtime_by_model_uniform.png

Usage (from repo root):
  python tools/ch6_perf_baseline_uniform.py --out figures/chapter_05/section6.4

Recommended:
  python tools/ch6_perf_baseline_uniform.py --out figures/chapter_05/section6.4 \
      --method DOP853 --rtol 1e-6 --atol 1e-9 --T 10 --fps 200 --repeats 5

Notes:
- Includes inverted pendulum open-loop and (optionally) closed-loop (swing-up + LQR).
  Disable closed-loop with: --no-inverted-closed
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def _wrap_repo_root_for_import() -> None:
    """Allow running from anywhere: add repo root (parent of 'dss/') to sys.path."""
    here = Path(__file__).resolve()
    cand = here.parent.parent
    if (cand / "dss").exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))


def _median(values) -> float:
    a = np.asarray(values, dtype=float)
    return float(np.median(a))


def _safe_int(x, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _extract_nfev(sol, diag) -> int:
    """Best-effort extraction of number of RHS evaluations."""
    if isinstance(diag, dict):
        for k in ("nfev", "n_fev", "rhs_evals", "fun_evals"):
            if k in diag:
                return _safe_int(diag[k])
    for k in ("nfev", "n_fev", "rhs_evals", "fun_evals"):
        if hasattr(diag, k):
            return _safe_int(getattr(diag, k))
    if hasattr(sol, "nfev"):
        return _safe_int(getattr(sol, "nfev"))
    return -1


@dataclass
class Case:
    name: str
    n: int
    build_system: Callable[[], object]
    y0: np.ndarray


def build_cases(include_inverted_closed: bool) -> list[Case]:
    def build_pendulum():
        from dss.models.pendulum import Pendulum
        return Pendulum(length=1.0, mass=1.0, mode="damped", damping=0.05)

    def build_double_pendulum():
        from dss.models.double_pendulum import DoublePendulum
        return DoublePendulum(length1=1.0, length2=1.0, mass1=1.0, mass2=1.0, mode="ideal")

    def build_dc_motor():
        from dss.models.dc_motor import DCMotor
        return DCMotor(
            R=1.0, L=0.5, Ke=0.1, Kt=0.1,
            J=0.02, bm=0.02,
            v_mode="step", V0=12.0, t_step=0.2,
            load_mode="none",
        )

    def build_vanderpol():
        from dss.models import VanDerPol
        return VanDerPol(mu=2.0)

    def build_lorenz():
        from dss.models.lorenz import Lorenz
        return Lorenz(sigma=10.0, rho=28.0, beta=8/3)

    def build_inverted_open():
        from dss.models.inverted_pendulum import InvertedPendulum
        return InvertedPendulum(
            mode="ideal",
            length=1.0,
            cart_mass=0.5,
            mass=0.2,
            gravity=9.81,
            coulomb_k=1000.0,
            b_cart=0.0, coulomb_cart=0.0,
            b_pend=0.0, coulomb_pend=0.0,
        )

    def build_inverted_closed():
        from dss.models.inverted_pendulum import InvertedPendulum
        from dss.controllers.swingup import AutoSwingUp
        from dss.controllers.lqr_controller import AutoLQR
        from dss.controllers.simple_switcher import SimpleSwitcher
        from dss.wrappers.closed_loop_cart import ClosedLoopCart

        plant = InvertedPendulum(
            mode="ideal",
            length=1.0,
            cart_mass=0.5,
            mass=0.2,
            gravity=9.81,
            coulomb_k=1000.0,
            b_cart=0.0, coulomb_cart=0.0,
            b_pend=0.0, coulomb_pend=0.0,
        )
        swing = AutoSwingUp(plant, ke=30.0, force_limit=30.0, du_max=800.0)
        Q = np.diag([10.0, 15.0, 120.0, 250.0])
        lqr = AutoLQR(plant, Q=Q, u_max=30.0)
        ctrl = SimpleSwitcher(
            system=plant,
            lqr_controller=lqr,
            swingup_controller=swing,
            engage_angle_deg=27.0,
            engage_speed_rad_s=9.0,
            engage_cart_speed=6.0,
            dropout_angle_deg=45.0,
            dropout_speed_rad_s=30.0,
            dropout_cart_speed=10.0,
            allow_dropout=True,
        )
        return ClosedLoopCart(plant, ctrl)

    cases = [
        Case("dc_motor", 2, build_dc_motor, np.array([0.0, 0.0], dtype=float)),
        Case("pendulum", 2, build_pendulum, np.array([0.8, 0.0], dtype=float)),
        Case("vanderpol", 2, build_vanderpol, np.array([0.1, 0.0], dtype=float)),
        Case("lorenz", 3, build_lorenz, np.array([1.0, 1.0, 1.0], dtype=float)),
        Case("double_pendulum", 4, build_double_pendulum, np.array([1.2, 0.0, 1.0, 0.0], dtype=float)),
        Case("inverted_open", 4, build_inverted_open, np.array([0.0, 0.0, 3.14, 0.0], dtype=float)),
    ]
    if include_inverted_closed:
        cases.append(Case("inverted_closed", 4, build_inverted_closed, np.array([0.0, 0.0, 3.14, 0.0], dtype=float)))
    return cases


def run_benchmark(out_dir: Path, T: float, fps: int, method: str, rtol: float, atol: float,
                  repeats: int, include_inverted_closed: bool) -> list[dict]:
    from dss.core.simulator import simulate

    cases = build_cases(include_inverted_closed)
    rows: list[dict] = []

    for case in cases:
        runtimes = []
        nfevs = []

        for _ in range(repeats):
            system = case.build_system()

            t0 = perf_counter()
            sol, diag = simulate(
                system,
                case.y0,
                T=float(T),
                fps=int(fps),
                method=str(method),
                rtol=float(rtol),
                atol=float(atol),
                return_diagnostics=True,
            )
            dt = perf_counter() - t0

            runtimes.append(dt)
            nfevs.append(_extract_nfev(sol, diag))

        row = {
            "Model": case.name,
            "n": case.n,
            "T": float(T),
            "fps": int(fps),
            "N": int(round(float(T) * int(fps))) + 1,
            "Method": method,
            "rtol": float(rtol),
            "atol": float(atol),
            "runtime_s_med": _median(runtimes),
            "nfev_med": int(np.median(np.asarray(nfevs, dtype=float))) if all(v >= 0 for v in nfevs) else -1,
        }
        rows.append(row)
        print(f"[OK] {case.name}: median {row['runtime_s_med']:.3e} s, nfev {row['nfev_med']}")

    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_tex_table(rows: list[dict], out_path: Path) -> None:
    lines = []
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\hline")
    lines.append(r"Model & $n$ & $T$ & $N$ & Method & rtol & atol & runtime [s] & nfev \\")
    lines.append(r"\hline")
    for r in rows:
        lines.append(
            f"{r['Model']} & {r['n']} & {r['T']:.0f} & {r['N']} & {r['Method']} & "
            f"{r['rtol']:.0e} & {r['atol']:.0e} & {r['runtime_s_med']:.3e} & {r['nfev_med']} \\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_runtime_plot(rows: list[dict], out_path: Path) -> None:
    names = [r["Model"] for r in rows]
    vals = [r["runtime_s_med"] for r in rows]

    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.bar(names, vals)
    ax.set_ylabel("runtime [s] (median)")
    ax.set_title("Uniform baseline runtime across models")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _wrap_repo_root_for_import()

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="Output directory (CSV + TeX + PNG).")
    p.add_argument("--T", type=float, default=10.0, help="Uniform simulation horizon for all models [s].")
    p.add_argument("--fps", type=int, default=200, help="Uniform sampling rate for all models [Hz].")
    p.add_argument("--method", type=str, default="DOP853")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-9)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--no-inverted-closed", action="store_true", help="Exclude inverted_closed case.")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_benchmark(
        out_dir=out_dir,
        T=args.T,
        fps=args.fps,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        repeats=args.repeats,
        include_inverted_closed=(not args.no_inverted_closed),
    )

    csv_path = out_dir / "perf_baseline_uniform.csv"
    tex_path = out_dir / "table_perf_baseline_uniform.tex"
    png_path = out_dir / "runtime_by_model_uniform.png"

    write_csv(rows, csv_path)
    write_tex_table(rows, tex_path)
    write_runtime_plot(rows, png_path)

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {tex_path}")
    print(f"[OK] wrote {png_path}")


if __name__ == "__main__":
    main()
