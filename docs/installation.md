# Installation

This project is intended to be installed in editable mode (recommended) so imports work consistently in Streamlit, scripts, and IDEs.

## Requirements

- Python 3.10+ recommended
- Dependencies are defined in `pyproject.toml` (legacy `requirements.txt` is kept for reference).

Install:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
# optional (dev tools): python -m pip install -e ".[dev]"
```

## Running the GUI

Run from the project root:

```bash
streamlit run apps/streamlit/app.py
```

## Running scripts

All scripts are plain Python files in `dss/scripts/`. Examples:

```bash
python dss/scripts/pendulum_small_ang.py
python dss/scripts/vdp_limit_cycle.py
python dss/scripts/lorenz_attractor.py
python dss/scripts/performance.py
```

Outputs (CSV and PNG figures) are written into `dss/scripts/data/` and `dss/scripts/figs/` for the scripts that generate Chapter-6-style artifacts.
