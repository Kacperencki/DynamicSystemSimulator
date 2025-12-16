# scripts/ch6_00_build_gallery_csv.py
#
# Build compact, consistently named "gallery" CSV files for Chapter 6
# time-domain figures from the detailed time series already produced by:
#   - ch6_01_pendulum_small_angle.py
#   - ch6_02_vdp_limit_cycle.py
#   - ch6_03_dc_motor_step.py
#   - ch6_04_inverted_case_study.py
#   - ch6_06_double_pendulum_chaos.py
#   - ch6_07_lorenz_attractor.py
#
# Outputs (all in data/):
#   ch6_gallery_pendulum.csv
#   ch6_gallery_double_pendulum.csv
#   ch6_gallery_vdp.csv
#   ch6_gallery_dc_motor.csv
#   ch6_gallery_lorenz.csv
#   ch6_gallery_inverted_openloop.csv
#   ch6_gallery_inverted_swingup.csv
#   ch6_gallery_inverted_lqr.csv

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


def _require_csv(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        print(f"[WARN] Missing input CSV: {path}", file=sys.stderr)
    return path


def build_pendulum_gallery() -> None:
    """t, theta, theta_dot from small-angle test."""
    src = _require_csv("ch6_pendulum_small_angle_timeseries.csv")
    if not src.exists():
        return
    df = pd.read_csv(src)
    out = df[["t", "theta", "theta_dot"]].copy()
    out.to_csv(DATA_DIR / "ch6_gallery_pendulum.csv", index=False)
    print("[OK] ch6_gallery_pendulum.csv")


def build_double_pendulum_gallery() -> None:
    """
    t, theta1, theta1_dot, theta2, theta2_dot from the A-branch of the chaos test.
    We take the 'A' trajectory as the representative one.
    """
    src = _require_csv("ch6_double_chaos_timeseries.csv")
    if not src.exists():
        return
    df = pd.read_csv(src)
    out = pd.DataFrame(
        {
            "t": df["t"],
            "theta1": df["theta1_A"],
            "theta1_dot": df["theta1_dot_A"],
            "theta2": df["theta2_A"],
            "theta2_dot": df["theta2_dot_A"],
        }
    )
    out.to_csv(DATA_DIR / "ch6_gallery_double_pendulum.csv", index=False)
    print("[OK] ch6_gallery_double_pendulum.csv")


def build_vdp_gallery(ic_index: int = 0) -> None:
    """
    Use one initial condition (default ic_index=0) from the
    Van der Pol limit-cycle script as the representative trajectory.
    """
    src = _require_csv("ch6_vdp_limit_cycle_timeseries.csv")
    if not src.exists():
        return
    df = pd.read_csv(src)
    df_ic = df[df["ic_index"] == ic_index].copy()
    if df_ic.empty:
        print(f"[WARN] No rows with ic_index={ic_index} in {src}", file=sys.stderr)
        return
    out = df_ic[["t", "v", "iL"]].copy()
    out.to_csv(DATA_DIR / "ch6_gallery_vdp.csv", index=False)
    print("[OK] ch6_gallery_vdp.csv (ic_index =", ic_index, ")")


def build_dc_motor_gallery() -> None:
    """t, i, omega from DC motor step response."""
    src = _require_csv("ch6_dc_motor_step_timeseries.csv")
    if not src.exists():
        return
    df = pd.read_csv(src)
    out = df[["t", "i", "omega"]].copy()
    out.to_csv(DATA_DIR / "ch6_gallery_dc_motor.csv", index=False)
    print("[OK] ch6_gallery_dc_motor.csv")


def build_lorenz_gallery() -> None:
    """t, x, y, z from Lorenz A-branch."""
    src = _require_csv("ch6_lorenz_timeseries.csv")
    if not src.exists():
        return
    df = pd.read_csv(src)
    out = df[["t", "x", "y", "z"]].copy()
    out.to_csv(DATA_DIR / "ch6_gallery_lorenz.csv", index=False)
    print("[OK] ch6_gallery_lorenz.csv")


def build_inverted_openloop_gallery() -> None:
    """t, x, theta from inverted pendulum open-loop case."""
    src = _require_csv("ch6_inverted_openloop_timeseries.csv")
    if not src.exists():
        return
    df = pd.read_csv(src)
    out = df[["t", "x", "theta"]].copy()
    out.to_csv(DATA_DIR / "ch6_gallery_inverted_openloop.csv", index=False)
    print("[OK] ch6_gallery_inverted_openloop.csv")


def build_inverted_swingup_gallery() -> None:
    """t, x, theta from swing-up case study."""
    src = _require_csv("ch6_inverted_swingup_timeseries.csv")
    if not src.exists():
        return
    df = pd.read_csv(src)
    out = df[["t", "x", "theta"]].copy()
    out.to_csv(DATA_DIR / "ch6_gallery_inverted_swingup.csv", index=False)
    print("[OK] ch6_gallery_inverted_swingup.csv")


def build_inverted_lqr_gallery() -> None:
    """t, x, theta from LQR stabilisation case study."""
    src = _require_csv("ch6_inverted_lqr_timeseries.csv")
    if not src.exists():
        return
    df = pd.read_csv(src)
    out = df[["t", "x", "theta"]].copy()
    out.to_csv(DATA_DIR / "ch6_gallery_inverted_lqr.csv", index=False)
    print("[OK] ch6_gallery_inverted_lqr.csv")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    build_pendulum_gallery()
    build_double_pendulum_gallery()
    build_vdp_gallery(ic_index=0)
    build_dc_motor_gallery()
    build_lorenz_gallery()
    build_inverted_openloop_gallery()
    build_inverted_swingup_gallery()
    build_inverted_lqr_gallery()


if __name__ == "__main__":
    main()
