# Contributing

## Coding style

- Keep numerical code in `dss/` independent from Streamlit.
- Keep Streamlit code in `apps/streamlit/` limited to UI + plotting.
- Prefer pure functions for dashboards and runners so they are easy to test and reuse.

## Adding a new system

See `docs/extending.md`.

## Common patterns

- Model: implements `dynamics`, `state_labels`, (optional) `positions`.
- Runner: converts UI values → model instance → `Solver` → `cfg/out`.
- Dashboard: takes `cfg/out` and returns a single Plotly `Figure`.
- Registry: imports `get_spec()` and registers into `SYSTEM_SPECS`.
