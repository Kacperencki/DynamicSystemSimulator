# Development

This page documents recommended development practices for DSS.

## Project goals (engineering)

- Keep the **numerical core** independent of Streamlit.
- Keep Streamlit code focused on UI, plotting, and orchestration.
- Make it easy to add new systems without editing unrelated layers.

## Local development setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS/Linux

python -m pip install -e ".[gui,dev]"
```

## Linting (Ruff)

Ruff is configured in `pyproject.toml`.

Run:

```bash
ruff check .
```

The current configuration focuses on real errors (undefined names, etc.). You can gradually expand the rule set once the codebase is stable.

## Testing (pytest)

Tests live in `tests/`.

Run:

```bash
pytest
```

Recommended minimal test coverage:
- each model: dynamics returns correct shape and finite values
- solver: a short integration run produces consistent `(T, X)`
- wrappers/controllers: closed-loop runs do not produce NaNs
- (optional) logger: produces expected artifact files

## Documentation (MkDocs)

Serve docs locally:

```bash
mkdocs serve
```

## Keeping the repository clean

Do not commit generated files:
- `__pycache__/`, `*.pyc`
- `.venv/`
- `artifacts/`
- `site/` (MkDocs output)
- `.pytest_cache/`, `.ruff_cache/`

Use the cleanup scripts under `tools/` to remove common generated directories.

## Making changes safely

Recommended workflow for changes that affect simulation behavior:

1. Add or update a test that describes the expected behavior.
2. Implement the change.
3. Run `pytest`.
4. Run a short GUI check in Streamlit for at least one system.

This keeps the “thesis promises” (modularity, testability, reproducibility) aligned with the implementation.
