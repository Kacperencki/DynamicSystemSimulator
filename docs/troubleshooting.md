# Troubleshooting

## `ModuleNotFoundError: No module named 'apps'`

Cause:
- running Streamlit from the wrong working directory (not the project root).

Fix:
- run from the folder that contains `apps/` and `dss/`:

```bash
cd path/to/project_root
streamlit run apps/streamlit/app.py
```

The entrypoint already inserts the project root into `sys.path`, but it relies on the correct file location.

## Plots look “cut off” / not full width in Streamlit

The GUI uses two columns (controls + main). If the right column is too narrow, adjust:

- column ratios in `apps/streamlit/app.py`:

```python
col_controls, col_main = st.columns([0.24, 0.76], gap="large")
```

Also check custom CSS in `apps/streamlit/assets/style.css`.

## NaN/Inf error during simulation

`Solver` raises `FloatingPointError` if NaN/Inf occurs.

Common causes:
- too aggressive parameters (large forcing, very low damping),
- unstable equilibria (inverted pendulum),
- too coarse output sampling combined with stiff dynamics (try smaller `dt` / larger `fps`),
- controller gains too high.

Try:
- reduce forcing amplitude,
- increase damping,
- use tighter tolerances (`rtol`, `atol`) or a different method (e.g., `Radau` for stiff systems),
- shorten `T` to localize the issue.

## `dss/core/simulator.py` import error (`vizualizer`)

That file references `vizualizer.visualizer.MatplotlibVisualizer`, which is not included.
Use:
- Streamlit GUI, or
- direct solver usage (`dss/core/solver.py`).
