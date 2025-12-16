# Dynamic System Simulator (DSS)

DSS is a small Python toolkit for simulating classic dynamical systems (mechanical + electrical) and exploring their trajectories through an interactive **Streamlit + Plotly** GUI.

The repository has two main parts:

- `dss/`: the core simulation library (models, solver, controllers, wrappers, logging, reproducibility scripts)
- `apps/streamlit/`: the Streamlit application (controls, runners, dashboards, registry)

## Systems included

- Single pendulum (ideal / damped / driven)
- Double pendulum (including chaotic demos)
- Inverted pendulum on a cart (open-loop + closed-loop)
- Van der Pol oscillator (circuit form)
- Lorenz system
- DC motor (electromechanical model)

## Quick start

### 1) Install dependencies

Create and activate a virtual environment, then install:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Run the Streamlit GUI

Run from the project root:

```bash
streamlit run apps/streamlit/app.py
```

If you see `ModuleNotFoundError: No module named 'apps'`, ensure you are running the command from the project root (the folder that contains `apps/` and `dss/`). See `docs/troubleshooting.md`.

## Core usage (without Streamlit)

Minimal example using the solver directly:

```python
import numpy as np
from dss.models.pendulum import Pendulum
from dss.core.solver import Solver

system = Pendulum(length=1.0, mass=1.0, mode="damped", damping=0.02, gravity=9.81)
x0 = np.array([0.6, 0.0])  # [theta, theta_dot]

sol = Solver(system, x0, T=10.0, fps=200, method="RK45").run()

t = sol.t              # shape: (N,)
X = sol.y.T            # shape: (N, n_state)
```

## Project layout

```text
dss_proj/
├── apps/
│   ├── __init__.py
│   └── streamlit/
│       ├── __init__.py
│       ├── app.py
│       ├── assets/
│       │   └── style.css
│       ├── components/
│       │   ├── animations.py
│       │   ├── controls_common.py
│       │   ├── dashboards/
│       │   └── plots_view.py
│       ├── layout.py
│       ├── registry.py
│       ├── runners/
│       │   ├── dc_motor_runner.py
│       │   ├── inverted_runner.py
│       │   ├── lorenz_runner.py
│       │   ├── pendulum_runner.py
│       │   └── vanderpol_runner.py
│       └── systems/
│           ├── dc_motor_view.py
│           ├── double_pendulum_view.py
│           ├── inverted_pendulum_view.py
│           ├── lorenz_view.py
│           ├── single_pendulum_view.py
│           └── vanderpol_view.py
├── docs/
└── dss/
    ├── __init__.py
    ├── controllers/
    │   ├── __init__.py
    │   ├── linearize.py
    │   ├── lqr_controller.py
    │   ├── simple_switcher.py
    │   ├── swingup.py
    │   └── switcher.py
    ├── core/
    │   ├── __init__.py
    │   ├── experiments.py
    │   ├── logger.py
    │   ├── presets.py
    │   ├── simulator.py
    │   └── solver.py
    ├── models/
    │   ├── __init__.py
    │   ├── dc_motor.py
    │   ├── double_pendulum.py
    │   ├── inverted_pendulum.py
    │   ├── lorenz.py
    │   ├── pendulum.py
    │   └── vanderpoll_circuit.py
    ├── scripts/
    │   ├── data/
    │   │   ├── ch6_dc_motor_step_summary.csv
    │   │   ├── ch6_dc_motor_step_timeseries.csv
    │   │   ├── ch6_double_chaos_summary.csv
    │   │   ├── ch6_double_chaos_timeseries.csv
    │   │   ├── ch6_gallery_dc_motor.csv
    │   │   ├── ch6_gallery_double_pendulum.csv
    │   │   ├── ch6_gallery_inverted_lqr.csv
    │   │   ├── ch6_gallery_inverted_swingup.csv
    │   │   ├── ch6_gallery_lorenz.csv
    │   │   ├── ch6_gallery_pendulum.csv
    │   │   ├── ch6_gallery_vdp.csv
    │   │   ├── ch6_inverted_lqr_timeseries.csv
    │   │   ├── ch6_inverted_summary.csv
    │   │   ├── ch6_inverted_swingup_timeseries.csv
    │   │   ├── ch6_lorenz_summary.csv
    │   │   ├── ch6_lorenz_timeseries.csv
    │   │   ├── ch6_pendulum_small_angle_summary.csv
    │   │   ├── ch6_pendulum_small_angle_timeseries.csv
    │   │   ├── ch6_vdp_limit_cycle_summary.csv
    │   │   └── ch6_vdp_limit_cycle_timeseries.csv
    │   ├── dc_motor_step.py
    │   ├── double_p_chaos.py
    │   ├── figs/
    │   │   ├── ch6_dc_motor_step_response.png
    │   │   ├── ch6_double_angle_difference.png
    │   │   ├── ch6_double_energy_drift.png
    │   │   ├── ch6_inverted_lqr_theta.png
    │   │   ├── ch6_inverted_swingup_theta.png
    │   │   ├── ch6_lorenz_attractor_3d.png
    │   │   ├── ch6_lorenz_difference.png
    │   │   ├── ch6_pendulum_energy_drift.png
    │   │   ├── ch6_pendulum_theta.png
    │   │   ├── ch6_runtime_per_model.png
    │   │   ├── ch6_runtime_vs_tol_pendulum.png
    │   │   ├── ch6_vdp_phase_portrait.png
    │   │   └── ch6_vdp_time_series.png
    │   ├── gallery.py
    │   ├── inverted_case_study.py
    │   ├── logs/
    │   │   └── runs.jsonl
    │   ├── lorenz_attractor.py
    │   ├── pendulum_small_ang.py
    │   ├── performance.py
    │   └── vdp_limit_cycle.py
    └── wrappers/
        ├── __init__.py
        ├── closed_lood_cart.py
        └── motor_wrapper.py
```

## Documentation

Start here:

- `docs/overview.md` – what DSS is and how it is organized
- `docs/installation.md` – setup instructions
- `docs/streamlit_gui.md` – how the GUI is structured (SystemSpec pattern)
- `docs/models.md` – state definitions, parameters, and conventions per model
- `docs/core_api.md` – solver, diagnostics, logger
- `docs/extending.md` – how to add a new model end-to-end
- `docs/scripts.md` – scripts used to generate data/figures (e.g., for Chapter 6)
- `docs/troubleshooting.md` – common issues and fixes

## Notes

- The Streamlit app adds the project root to `sys.path` so it can be run without packaging. For production/CI, consider adding a proper `pyproject.toml` and installing with `pip install -e .`.
- `dss/core/simulator.py` references `vizualizer.visualizer.MatplotlibVisualizer`, which is **not included** in this repository. The modern path is the Streamlit GUI or direct solver usage.

