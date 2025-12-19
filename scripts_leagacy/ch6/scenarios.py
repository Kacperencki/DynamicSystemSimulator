"""Scenario registry for Chapter 6.

What this is
------------
A single, explicit list of "Chapter 6 experiments" with:
- which script produces the figure/table,
- default simulation settings (T, fps, method, rtol, atol),
- expected outputs (data/figs files),
- and where it will be referenced in the thesis (section id).

Why this exists
---------------
Chapter 6 currently uses multiple standalone scripts (pendulum, Van der Pol,
Lorenz, etc.). The registry is the foundation for a one-command pipeline
that can regenerate the entire chapter.

In this first migration step, each registry entry points to a *legacy*
producer script in `scripts/*.py`. Later, those producers will be refactored
into callable functions that consume `ScenarioSpec` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class ScenarioSpec:
    """Definition of one Chapter 6 experiment."""

    key: str
    section: str
    title: str
    description: str

    # Producer: for now, a path to an existing script under `scripts/`.
    producer_script: str

    # Default solver/simulation settings used by this scenario.
    # NOTE: legacy scripts may override these internally; during migration
    # we will make them read these defaults from the registry.
    defaults: Dict[str, Any] = field(default_factory=dict)

    # Expected outputs (relative paths). These are used for a "manifest".
    outputs: List[str] = field(default_factory=list)

    # Optional tags to enable filtered runs.
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry entries (current producers = existing scripts/*.py)
# ---------------------------------------------------------------------------

SCENARIOS: Mapping[str, ScenarioSpec] = {
    "behaviour_panels": ScenarioSpec(
        key="behaviour_panels",
        section="6.2",
        title="Functional behaviour panels (all models)",
        description=(
            "One-command producer for the Chapter 6.2 gallery: runs the key models "
            "(aligned with Chapter 4 order) and exports consistent panels + CSVs into "
            "artifacts/ch6/."
        ),
        producer_script="ch6/behaviour_panels.py",
        defaults={"method": "RK45"},
        outputs=[
            "artifacts/ch6/figs/ch6_behaviour_pendulum.png",
            "artifacts/ch6/figs/ch6_behaviour_double_pendulum.png",
            "artifacts/ch6/figs/ch6_behaviour_inverted_pendulum.png",
            "artifacts/ch6/figs/ch6_behaviour_dc_motor.png",
            "artifacts/ch6/figs/ch6_behaviour_vanderpol.png",
            "artifacts/ch6/figs/ch6_behaviour_lorenz.png",
            "artifacts/ch6/data/ch6_behaviour_pendulum.csv",
            "artifacts/ch6/data/ch6_behaviour_double_pendulum.csv",
            "artifacts/ch6/data/ch6_behaviour_inverted_swingup.csv",
            "artifacts/ch6/data/ch6_behaviour_inverted_lqr.csv",
            "artifacts/ch6/data/ch6_behaviour_dc_motor.csv",
            "artifacts/ch6/data/ch6_behaviour_vanderpol.csv",
            "artifacts/ch6/data/ch6_behaviour_lorenz.csv",
            "artifacts/ch6/logs/ch6_behaviour_manifest.json",
        ],
        tags=["gallery", "functional"],
    ),
    "pendulum_small_angle": ScenarioSpec(
        key="pendulum_small_angle",
        section="6.2",
        title="Single pendulum: small-angle behaviour + energy drift",
        description=(
            "Validates the ideal pendulum in the linear regime and checks energy "
            "conservation; outputs theta(t) and E(t)-E(0)."
        ),
        producer_script="pendulum_small_ang.py",
        defaults={"T": 10.0, "fps": 200, "method": "RK45", "rtol": 1e-6, "atol": 1e-8},
        outputs=[
            "data/ch6_pendulum_small_angle_summary.csv",
            "data/ch6_pendulum_small_angle_timeseries.csv",
            "figs/ch6_pendulum_theta.png",
            "figs/ch6_pendulum_energy_drift.png",
        ],
        tags=["gallery", "functional", "energy"],
    ),
    "vdp_limit_cycle": ScenarioSpec(
        key="vdp_limit_cycle",
        section="6.2",
        title="Van der Pol circuit: limit cycle",
        description=(
            "Demonstrates self-sustained oscillations and the phase portrait for the "
            "Van der Pol oscillator circuit model."
        ),
        producer_script="vdp_limit_cycle.py",
        defaults={"T": 30.0, "fps": 400, "method": "RK45", "rtol": 1e-6, "atol": 1e-8},
        outputs=[
            "data/ch6_vdp_limit_cycle_summary.csv",
            "data/ch6_vdp_limit_cycle_timeseries.csv",
            "figs/ch6_vdp_time_series.png",
            "figs/ch6_vdp_phase_portrait.png",
        ],
        tags=["gallery", "functional", "nonlinear"],
    ),
    "dc_motor_step": ScenarioSpec(
        key="dc_motor_step",
        section="6.2",
        title="DC motor: step response",
        description=(
            "Applies a voltage step and records current i(t) and angular speed ω(t)."
        ),
        producer_script="dc_motor_step.py",
        defaults={"T": 2.0, "fps": 2000, "method": "RK45", "rtol": 1e-6, "atol": 1e-8},
        outputs=[
            "data/ch6_dc_motor_step_summary.csv",
            "data/ch6_dc_motor_step_timeseries.csv",
            "figs/ch6_dc_motor_step_response.png",
        ],
        tags=["gallery", "functional", "electrical"],
    ),
    "inverted_case_study": ScenarioSpec(
        key="inverted_case_study",
        section="6.2",
        title="Inverted pendulum: swing-up and LQR",
        description=(
            "Case study demonstrating swing-up + stabilisation; produces θ(t) plots and "
            "time series for later analysis."
        ),
        producer_script="inverted_case_study.py",
        defaults={"T": 10.0, "fps": 200, "method": "RK45", "rtol": 1e-5, "atol": 1e-7},
        outputs=[
            "artifacts/ch6/data/ch6_convergence_summary.csv",
            "artifacts/ch6/figs/ch6_convergence_residuals.png",
            "artifacts/ch6/figs/ch6_convergence_tradeoffs.png",
            "artifacts/ch6/logs/ch6_convergence_manifest.json",
        ],

        tags=["gallery", "control"],
    ),
    "double_pendulum_chaos": ScenarioSpec(
        key="double_pendulum_chaos",
        section="6.2",
        title="Double pendulum: sensitivity to initial conditions",
        description=(
            "Runs two nearly-identical initial conditions and tracks divergence "
            "(a simple chaos/sensitivity demonstration)."
        ),
        producer_script="double_p_chaos.py",
        defaults={"T": 20.0, "fps": 200, "method": "RK45", "rtol": 1e-6, "atol": 1e-8},
        outputs=[
            "data/ch6_double_chaos_summary.csv",
            "data/ch6_double_chaos_timeseries.csv",
            "figs/ch6_double_angle_difference.png",
            "figs/ch6_double_energy_drift.png",
        ],
        tags=["gallery", "functional", "nonlinear"],
    ),
    "lorenz_attractor": ScenarioSpec(
        key="lorenz_attractor",
        section="6.2",
        title="Lorenz system: chaotic attractor",
        description=(
            "Generates Lorenz trajectories and visualises 3D attractor and divergence." 
            "In convergence studies, horizon is typically shortened (e.g. 5 s)."
        ),
        producer_script="lorenz_attractor.py",
        defaults={"T": 10.0, "fps": 200, "method": "RK45", "rtol": 1e-6, "atol": 1e-9},
        outputs=[
            "data/ch6_lorenz_summary.csv",
            "data/ch6_lorenz_timeseries.csv",
            "figs/ch6_lorenz_attractor_3d.png",
            "figs/ch6_lorenz_difference.png",
        ],
        tags=["gallery", "functional", "chaotic"],
    ),
    "performance": ScenarioSpec(
        key="performance",
        section="6.4",
        title="Performance benchmarks",
        description=(
            "Benchmarks runtime per model (common tolerances) and runtime vs tolerance "
            "for the pendulum."
        ),
        producer_script="performance.py",
        defaults={"T": 10.0, "fps": 200, "method": "RK45", "rtol": 1e-4, "atol": 1e-6},
        outputs=[
            "data/ch6_performance_models.csv",
            "data/ch6_performance_pendulum_tol.csv",
            "figs/ch6_runtime_per_model.png",
            "figs/ch6_runtime_vs_tol_pendulum.png",
        ],
        tags=["performance"],
    ),
    "gallery_csv": ScenarioSpec(
        key="gallery_csv",
        section="6.2",
        title="Build compact gallery CSVs",
        description=(
            "Post-process detailed time-series CSVs into compact gallery CSVs used by "
            "LaTeX figures or external plotting."
        ),
        producer_script="gallery.py",
        defaults={},
        outputs=[
            "data/ch6_gallery_pendulum.csv",
            "data/ch6_gallery_double_pendulum.csv",
            "data/ch6_gallery_vdp.csv",
            "data/ch6_gallery_dc_motor.csv",
            "data/ch6_gallery_lorenz.csv",
            "data/ch6_gallery_inverted_swingup.csv",
            "data/ch6_gallery_inverted_lqr.csv",
        ],
        tags=["gallery"],
    ),
    "benchmarks_suite": ScenarioSpec(
        key="benchmarks_suite",
        section="6.4",
        title="Runtime benchmarks across models and solvers",
        description="Benchmarks runtime (and nfev) per model/solver under consistent settings.",
        producer_script="ch6/benchmarks_suite.py",
        defaults={"T": 10.0, "rtol": 1e-4, "atol": 1e-6, "repeats": 3},
        outputs=[
            "artifacts/ch6/data/ch6_benchmarks_runtime.csv",
            "artifacts/ch6/figs/ch6_benchmarks_runtime.png",
            "artifacts/ch6/figs/ch6_benchmarks_nfev.png",
            "artifacts/ch6/logs/ch6_benchmarks_manifest.json",
        ],
        tags=["performance"],
    ),


    # -------------------------------------------------------------------
    # Placeholders for the next migration steps (not implemented yet).
    # These will become new producers under scripts/ch6/.
    # -------------------------------------------------------------------
    "convergence_sweeps": ScenarioSpec(
        key="convergence_sweeps",
        section="6.3",
        title="Solver convergence sweeps (all models)",
        description=(
            "Compute error metrics against a strict reference solution using a fixed "
            "evaluation grid, then report e(t), e_max, e_RMS vs rtol and method."
        ),
        producer_script="ch6/convergence_sweeps.py",
        defaults={"rtol_grid": [1e-2, 1e-4, 1e-6, 1e-8], "reference": "DOP853 @ 1e-12"},
        outputs=[
            "artifacts/ch6/data/ch6_convergence_summary.csv",
            "artifacts/ch6/figs/ch6_convergence_residuals.png",
        ],
        tags=["convergence"],
    ),
    "gui_timings": ScenarioSpec(
        key="gui_timings",
        section="6.6",
        title="Streamlit responsiveness benchmark",
        description=(
            "Measure simulation time vs plot-build vs serialisation time and approximate "
            "payload size per dashboard scenario."
        ),
        producer_script="ch6/gui_timings_report.py",
        defaults={"repeats": 10, "rtol": 1e-4, "T": 10.0},
        outputs=[
            "artifacts/ch6/data/ch6_gui_timings.csv",
            "artifacts/ch6/figs/ch6_gui_timings_breakdown.png",
        ],
        tags=["gui", "performance"],
    ),
}


RUN_ORDER = [
    "behaviour_panels",
    "pendulum_small_angle",
    "vdp_limit_cycle",
    "dc_motor_step",
    "inverted_case_study",
    "double_pendulum_chaos",
    "lorenz_attractor",
    "convergence_sweeps",
    "benchmarks_suite",
    "gallery_csv",
]

