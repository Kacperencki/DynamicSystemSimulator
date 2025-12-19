# Dynamic System Simulator (DSS)

Dynamic System Simulator (DSS) is a Python project for simulating classic dynamical systems (mechanical, electrical, and electromechanical) with an interactive **Streamlit + Plotly** GUI and a small, reusable numerical core.

## Key features

- Multiple built-in systems: single/double pendulum, inverted pendulum (open/closed loop), Lorenz, Van der Pol (circuit form), DC motor
- Thin solver wrapper built on `scipy.integrate.solve_ivp` (RK45, DOP853, Radau, BDF, …)
- Consistent, extensible Streamlit GUI based on a **SystemSpec** pattern (controls → run → dashboard)
- Optional logging utilities for reproducible runs (config + numerical output bundles)
- Offline tooling for generating figures/tables (e.g., thesis Chapter 6)

## Repository layout

- `dss/` — core library (models, solver/simulator, controllers, wrappers, logger)
- `apps/streamlit/` — Streamlit UI (registry, controls, dashboards, shared UI components)
- `tools/` — current, supported offline scripts (e.g., Chapter 6 figure/table generators)
- `scripts_leagacy/` — older offline scripts kept for reference (not guaranteed to match the current API)
- `figures/` — generated figures (recommended output target)
- `docs/` + `mkdocs.yml` — documentation (MkDocs)

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
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Streamlit GUI

From the project root:

```bash
streamlit run streamlit_app.py
```

## Offline tools (thesis figures/tables)

Run from the repository root.

### Chapter 6.2: model “diagnostic cards” (group figures)

```bash
python tools/ch6_generate_62.py \
  --out figures/chapter_05/section6.2 \
  --method DOP853 --rtol 1e-4 --atol 1e-6 --T 10 --fps 200
```

### Chapter 6.4: uniform performance baseline (runtime + nfev)

```bash
python tools/ch6_perf_baseline_uniform.py \
  --out figures/chapter_05/section6.4 \
  --method DOP853 --rtol 1e-4 --atol 1e-7 --T 10 --fps 200 --repeats 5
```

## Documentation (MkDocs)

```bash
mkdocs serve
```

## Where to start

- `docs/overview.md` — architecture and data flow
- `docs/streamlit_gui.md` — SystemSpec structure and how systems are wired
- `docs/extending.md` — how to add a new model end-to-end
- `docs/scripts.md` — offline tooling and artifact conventions
