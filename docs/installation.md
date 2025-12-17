# Installation

## Requirements

- Python **3.10+**
- Recommended: virtual environment (`venv`)

## Install (editable mode)

Editable installation is recommended for:
- consistent imports in IDEs,
- Streamlit running from the project root,
- development iteration.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

## Install GUI and development tools

```bash
python -m pip install -e ".[gui,dev]"
```

This installs:
- Streamlit (GUI)
- pytest (tests)
- ruff (lint)
- mkdocs (documentation)

## Run the GUI

From the project root:

```bash
streamlit run streamlit_app.py
```

If you run Streamlit from a different working directory, Python imports may fail. Always launch from the repository root unless you know exactly what you are doing.

## Run scripts

Scripts are intentionally separated from the `dss/` package. Run them from the repository root:

```bash
python scripts/lorenz_attractor.py
```

Outputs should go to `artifacts/`.

## Build documentation (MkDocs)

```bash
mkdocs serve
```

MkDocs will print a local URL you can open in your browser.

## Common setup problems

### `ModuleNotFoundError: No module named 'apps'` or `'dss'`

This typically happens when:
- you did not install the package in editable mode, or
- you are running Streamlit from a directory other than the project root.

Fix:
1. Activate your venv.
2. Run:

```bash
python -m pip install -e .
```

3. Launch Streamlit from the project root:

```bash
streamlit run streamlit_app.py
```

### Streamlit charts look “cut off” / not full width

This is usually caused by layout column ratios. In DSS, the main layout is defined in `streamlit_app.py` using `st.columns([...])`. If the right column is too narrow, increase its ratio.

See also: `docs/streamlit_gui.md`.
