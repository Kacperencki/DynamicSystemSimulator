# Reproducibility and logging

DSS supports a lightweight reproducibility workflow:

- store the exact parameters used for a run (`config.json`)
- store numerical output arrays (`output.npz`)
- optionally append a summary entry (`runs.jsonl`)

This is most useful for:
- experiments / validation chapters,
- generating plots and tables from fixed configurations,
- comparing solver settings across systems.

## Recommended output layout

A typical run directory looks like:

```
artifacts/
  logs/
    2025-12-17_21-30-10_pendulum_damped/
      config.json
      output.npz
      meta.json
```

### `config.json`

The configuration file should be JSON-serializable and contain:
- model name and parameters
- initial conditions
- solver method and tolerances
- any controller/wrapper settings (if used)

### `output.npz`

The numeric bundle typically contains:
- `T` (N,)
- `X` (N, n_state)
- optionally `U` (N,) or (N, n_u) if control signals are recorded

## How to use logging

### From scripts

A script can build a config and call a helper that saves the bundle. If you already have `cfg` and arrays:

```python
from dss.core.logger import save_run_bundle

save_run_bundle(
    base_dir="artifacts/logs",
    run_name="lorenz_demo",
    cfg=cfg,
    arrays={"T": T, "X": X},
)
```

### From the GUI

The Streamlit GUI is designed for interactive exploration. Logging can be added as an opt-in feature (e.g., a “Save run” checkbox) without changing simulation logic:
- the runner still returns `(cfg, out)`,
- logging writes a bundle after the run completes.

## What “reproducible” means here

Given a saved config and output bundle:
- the config can be rerun to regenerate the same type of trajectory,
- minor numeric differences can occur if you change:
  - solver method,
  - tolerances,
  - SciPy version,
  - floating-point environment.

For publication-quality comparisons, keep:
- Python version,
- SciPy version,
- solver settings,
as stable as practical.
