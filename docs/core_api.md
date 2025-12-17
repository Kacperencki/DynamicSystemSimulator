# Core API

This page describes the numerical core under `dss/core/`:

- `Solver`: wrapper around `scipy.integrate.solve_ivp`
- `simulate`: convenience API (if used)
- diagnostics helpers (where available)
- logging utilities (`dss/core/logger.py`)

## Solver (`dss/core/solver.py`)

### Purpose

`Solver` is a thin wrapper that standardizes how DSS runs `solve_ivp`:

- builds a sampling grid (`t_eval`) from `T` and `fps` (or uses user-provided `t_eval`),
- passes tolerances (`rtol`, `atol`) and method (`RK45`, `DOP853`, `Radau`, `BDF`, …),
- returns SciPy’s `OdeResult`.

### Typical usage

```python
import numpy as np
from dss.models.pendulum import Pendulum
from dss.core.solver import Solver

sys = Pendulum(length=1.0, mass=1.0, mode="damped", damping=0.02, coulomb=0.01)
y0 = np.array([0.4, 0.0])

sol = Solver(sys, initial_conditions=y0, T=10.0, fps=200, method="RK45", rtol=1e-5, atol=1e-8).run()

T = sol.t          # (N,)
X = sol.y.T        # (N, n_state)
```

### Output sampling vs. integration accuracy

- `fps` controls the density of **output samples** (how many points you get in `sol.t`).
- The integrator internally chooses adaptive steps to satisfy `rtol` and `atol`.

If you set `fps` extremely high, you increase:
- memory usage,
- figure payload size (Streamlit),
- and post-processing time.

It does not necessarily increase integration accuracy.

### Choosing `rtol` and `atol`

SciPy uses a combined error test roughly based on:

\[
\frac{\lvert e_i \rvert}{\text{atol}_i + \text{rtol} \cdot \lvert y_i \rvert} \le 1
\]

- `rtol`: relative tolerance (scales with state magnitude)
- `atol`: absolute tolerance (minimum absolute accuracy)

Practical guidance:
- start with `rtol=1e-4`, `atol=1e-6` for interactive exploration,
- tighten to `rtol=1e-6`, `atol=1e-9` for validation figures,
- stiff systems: consider `Radau`/`BDF` and measure performance.

## `simulate` convenience API (`dss/core/simulator.py`)

`simulate` is a helper that:
- constructs a `Solver`,
- returns the `OdeResult`,
- optionally returns diagnostics in a lightweight dictionary.

If you need maximum control (custom `t_eval`, events, etc.), prefer using `Solver` directly.

## Logging (`dss/core/logger.py`)

The logger utilities are intended for reproducible runs:

- save a run configuration (`config.json`)
- save numeric arrays (`output.npz`)
- append a run index to `runs.jsonl` (optional)

Recommended directory layout:

- `artifacts/logs/<timestamp>_<name>/config.json`
- `artifacts/logs/<timestamp>_<name>/output.npz`

The Streamlit GUI can remain interactive-first; scripts and experiments typically enable logging by default.

See: `docs/reproducibility.md`.
