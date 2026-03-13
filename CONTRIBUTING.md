# Contributing

This repository mixes a numerical library (`dss/`) with an interactive GUI (`apps/streamlit/`).
Changes should preserve the boundary between these two layers.

## Getting started

Clone the repository and install in editable mode with development extras:

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[gui,dev]"
```

The `dev` extra adds `pytest`, `ruff`, and `mkdocs`.

## Architecture rules

- Keep all numerical code (`dss/` — models, solver, controllers, wrappers, logging) **independent of Streamlit**.
- Keep Streamlit code limited to widget controls, runner orchestration, and Plotly dashboards/animations.
- Prefer small, composable functions over large monolithic files.

## Coding style

- Use clear names for physical quantities (`theta`, `omega`, `cart_pos`, …).
- Add type hints to all public interfaces.
- Keep plotting code in the GUI layer or in offline tooling under `tools/`.
- Document public classes and functions with docstrings. For physics or control logic include the relevant equation or cite where it comes from.
- Explain non-obvious constants and magic numbers with inline comments.

## Running tests

The test suite lives in `tests/` and covers models and controllers:

```bash
python -m pytest
```

All tests must pass before submitting a change. If you modify model equations, solver behaviour, or controller logic, add or update the relevant tests.

## Linting

```bash
ruff check .
```

Fix any reported issues before committing.

## Validating numerical changes

If a change affects model equations or solver behaviour, validate it with:

- `python -m pytest` — confirm all unit tests pass.
- A short GUI run (`streamlit run streamlit_app.py`) — pick one system and confirm plots look reasonable.
- An offline tool run to catch obvious regressions:

```bash
python tools/ch6_perf_baseline_uniform.py --out /tmp/check --method DOP853 --rtol 1e-4 --atol 1e-7 --T 10 --fps 200 --repeats 3
```

## Adding a new system

See `docs/extending.md` for a step-by-step guide.

## Documentation

Documentation is in `docs/` and built with MkDocs:

```bash
mkdocs serve
```

When changing public APIs or user-facing behaviour, update:
- `README.md`
- the relevant page under `docs/`

## Commit messages

Use short, direct commit messages. Prefix with the affected area when helpful:

```
fix(controllers): raise default engage_angle_deg
feat(models): add spring-mass system
docs: update installation instructions
```
