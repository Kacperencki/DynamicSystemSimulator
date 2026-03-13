# Dynamic System Simulator (DSS)

Dynamic System Simulator (DSS) is a Python project for simulating classic dynamical systems (mechanical, electrical, and electromechanical) with an interactive **Streamlit + Plotly** GUI and a small, reusable numerical core.

## Key features

- Six built-in systems: single/double pendulum, inverted pendulum (open/closed-loop with LQR, swing-up, and automatic handoff), Lorenz attractor, Van der Pol oscillator (circuit form), DC motor
- Thin solver wrapper built on `scipy.integrate.solve_ivp` (RK45, DOP853, Radau, BDF, …)
- Consistent, extensible Streamlit GUI based on a **SystemSpec** pattern (controls → run → dashboard)
- Every parameter widget carries a **?** help tooltip explaining what the variable means
- Optional logging utilities for reproducible runs (config + numerical output bundles)
- Offline tooling for generating thesis figures/tables (`tools/`)

## Repository layout

```
dss/                  Core library (models, solver, controllers, wrappers, logger)
  core/               Pipeline, solver, simulator, contracts, logging
  models/             ODE model classes (Pendulum, InvertedPendulum, DCMotor, …)
  controllers/        LQR, energy-based swing-up, supervisor switcher
  wrappers/           ClosedLoopCart, MotorWrapper
  utils/              Angle helpers, etc.

apps/streamlit/       Interactive GUI
  systems/            Per-system controls + view (one file per system)
  runners/            Bridge between UI values and DSS core
  components/         Reusable widgets, dashboards, animation builders
  layout.py           SystemSpec dataclass + generic render_system()
  registry.py         System catalog (display name → SystemSpec factory)

tools/                Supported offline scripts (Chapter 6 figure/table generators)
docs/                 MkDocs documentation
tests/                Unit tests
```

## Requirements

- Python **3.10+**
- Recommended: create and use a virtual environment

Dependencies are declared in `pyproject.toml`. A minimal `requirements.txt` is also provided.

## Installation

### Option A (recommended): editable install + GUI extras

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[gui]"
```

If you also want docs + lint/test tools:

```bash
python -m pip install -e ".[gui,dev]"
```

### Option B: requirements.txt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

python -m pip install -r requirements.txt
```

## Run the Streamlit GUI

```bash
streamlit run streamlit_app.py
```

Select a system from the left panel, adjust parameters (hover any **?** icon for a description), and click **Run**.

## Offline tools (thesis figures/tables)

Run from the repository root.

### Chapter 6.2 — model diagnostic cards

```bash
python tools/ch6_generate_62.py \
  --out figures/chapter_05/section6.2 \
  --method DOP853 --rtol 1e-4 --atol 1e-6 --T 10 --fps 200
```

### Chapter 6.4 — performance baseline

```bash
python tools/ch6_perf_baseline_uniform.py \
  --out figures/chapter_05/section6.4 \
  --method DOP853 --rtol 1e-4 --atol 1e-7 --T 10 --fps 200 --repeats 5
```

## Documentation (MkDocs)

```bash
mkdocs serve
```

## Where to start reading the code

| Goal | Start here |
|---|---|
| Architecture overview | `docs/overview.md` |
| Add a new system | `docs/extending.md` |
| Understand the GUI wiring | `docs/streamlit_gui.md` → `apps/streamlit/layout.py` |
| Understand the physics | `dss/models/<system>.py` (each has a module docstring) |
| Understand a controller | `dss/controllers/<name>.py` (module docstring with theory) |
| Run an ODE from Python | `dss/core/simulator.py` → `simulate()` |
| Full pipeline (config dict) | `dss/core/pipeline.py` → `run_config()` |
