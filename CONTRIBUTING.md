# Contributing

This repository mixes a numerical library (`dss/`) with an interactive GUI (`apps/streamlit/`). To keep the project maintainable, changes should preserve the boundary between these layers.

## Ground rules

- Keep numerical code (models, solver, controllers, wrappers, logging) **independent of Streamlit**.
- Keep Streamlit code limited to:
  - widget controls,
  - orchestration (calling runners),
  - Plotly dashboards/animations.
- Prefer small, composable functions over large monolithic files.

## Coding style

- Use clear names for physical quantities (`theta`, `omega`, `cart_pos`, …).
- Keep units consistent (SI).
- Avoid hidden global state.
- Return NumPy arrays with predictable shapes.

## Adding a new system

Follow `docs/extending.md`. High-level checklist:

1. Add model in `dss/models/` and implement:
   - `dynamics(t, state, inputs=None)`
   - `state_labels()`
2. Register it in `dss/models/__init__.py`.
3. Add a Streamlit `SystemSpec` under `apps/streamlit/systems/`.
4. Register the system factory in `apps/streamlit/registry.py`.
5. Add a minimal test in `tests/` (recommended).

## Tests

Run:

```bash
pytest
```

If a change modifies model equations or solver behavior, add a test that captures the intended behavior (shape checks, energy drift thresholds, etc.).

## Documentation

Documentation is in `docs/` and built with MkDocs:

```bash
mkdocs serve
```

When changing public APIs or user-facing behavior, update:
- README.md
- relevant docs pages under `docs/`
