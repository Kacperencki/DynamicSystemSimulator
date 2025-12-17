# Dynamic System Simulator (DSS)

Dynamic System Simulator (DSS) is a Python project for simulating classic dynamical systems (mechanical, electrical, and electromechanical) with an interactive **Streamlit + Plotly** GUI and a small, reusable numerical core.

## Key features

- Multiple built-in systems: pendulums (single/double), cart–pole, Lorenz, Van der Pol (circuit form), DC motor
- Thin solver wrapper built on `scipy.integrate.solve_ivp` (RK45, DOP853, Radau, BDF, …)
- Consistent, extensible Streamlit GUI based on a **SystemSpec** pattern (controls → run → dashboard)
- Optional logging utilities for reproducible runs (config + numerical output bundles)
- Scripts for generating figures and tables under `scripts/` (e.g., for experiments/validation)

## Repository layout

- `dss/` — core library (models, solver/simulator, controllers, wrappers, logger)
- `apps/streamlit/` — Streamlit UI (systems, runners, dashboards, shared UI components)
- `scripts/` — offline experiments (write outputs to `artifacts/`)
- `artifacts/` — generated outputs (not meant to be committed)
- `docs/` + `mkdocs.yml` — documentation (MkDocs)

## Requirements

- Python **3.10+**
- Recommended: create and use a virtual environment

Core dependencies are defined in `pyproject.toml`. A minimal `requirements.txt` is also provided.

## Installation

Editable installation is recommended so imports work consistently in IDEs and when launching Streamlit:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

If you plan to work on the GUI and docs:

```bash
python -m pip install -e ".[gui,dev]"
```

## Run the Streamlit GUI

From the project root:

```bash
streamlit run streamlit_app.py
```

## Run scripts (offline artifacts)

Scripts live in `scripts/` and should write outputs into `artifacts/`:

```bash
python scripts/pendulum_small_ang.py
python scripts/double_p_chaos.py
python scripts/vdp_limit_cycle.py
python scripts/lorenz_attractor.py
python scripts/performance.py
```

## Documentation (MkDocs)

```bash
pip install -e ".[dev]"
mkdocs serve
```

Then open the local MkDocs site shown in the console output.

## Core conventions (important)

### Model interface
Every model exposes a continuous-time ODE of the form \(\dot{x} = f(t, x, u)\). In code, this is represented by:

- `dynamics(t: float, state: np.ndarray, inputs=None) -> np.ndarray`

The solver calls `dynamics(t, state)` (no inputs). Controllers/wrappers may pass `inputs=...` when a model supports external actuation.

### Runner output contract
Runners (Streamlit and scripts) should return:
- `cfg`: JSON-serializable configuration dictionary
- `out`: dictionary containing at least:
  - `T`: time array, shape `(N,)`
  - `X`: state array, shape `(N, n_state)`

Additional arrays (e.g., control `U`) can be included as extra keys.

## Cleaning generated files

Do not commit caches and generated outputs. Use the provided cleanup helpers:

- `tools/cleanup_repo.py`
- `tools/cleanup_repo.ps1`

They remove `__pycache__`, `*.pyc`, and common build/test caches.

## Where to start

- `docs/overview.md` — architecture and data flow
- `docs/streamlit_gui.md` — SystemSpec structure and how systems are wired
- `docs/extending.md` — how to add a new model end-to-end
