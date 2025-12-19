# Development

This page documents recommended development practices for DSS.

## Project goals (engineering)

- Keep the **numerical core** independent of Streamlit.
- Keep Streamlit code focused on UI, plotting, and orchestration.
- Make it easy to add new systems without editing unrelated layers.

## Local development setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[gui,dev]"
```

## Running lint and docs

```bash
ruff check .
mkdocs serve
```

## Making changes safely

When a change affects simulation behaviour, verify it in at least two ways:

1. Run the GUI for one representative system (quick sanity check):

```bash
streamlit run streamlit_app.py
```

2. Run an offline tool that exercises the solver and writes artifacts:

```bash
python tools/ch6_perf_baseline_uniform.py --out figures/_dev_perf_check
```

This keeps “thesis promises” (modularity, reproducibility) aligned with the implementation even when a full test suite is not present.
