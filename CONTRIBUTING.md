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
- Prefer type hints for public interfaces.
- Keep plotting code in the GUI layer (or in offline tooling under `tools/`).

## Validating changes

If a change modifies model equations or solver behaviour, validate it with:
- a short GUI run (pick one system and confirm plots look reasonable), and
- an offline tool run (e.g., `tools/ch6_perf_baseline_uniform.py`) to catch obvious regressions.

A dedicated `tests/` directory is optional; add it if you are introducing non-trivial numerical changes that should be guarded long-term.

## Documentation

Documentation is in `docs/` and built with MkDocs:

```bash
mkdocs serve
```

When changing public APIs or user-facing behaviour, update:
- `README.md`
- relevant docs pages under `docs/`
