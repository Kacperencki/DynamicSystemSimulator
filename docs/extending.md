# Extending DSS: adding a new model

This guide describes the end-to-end path to add a new system that appears in the Streamlit GUI.

## 1) Add (or import) a model in `dss/models/`

Create a file `dss/models/my_system.py` with:

- `class MySystem:`
- `__init__(...)` with physical parameters
- `dynamics(t, state, u=None)` returning `state_dot`
- `state_labels()` for plots
- optional: `positions(state)` if you want an animation

Recommendation: keep `dynamics()` pure and deterministic (no Streamlit inside).

## 2) Create a runner in `apps/streamlit/runners/`

Runners are responsible for:
- translating UI control values into model parameters,
- building `x0` initial conditions,
- choosing `T`, `dt` (or `fps`),
- calling `Solver(...)` and formatting outputs.

Example skeleton:

```python
import numpy as np
from dss.core.solver import Solver
from dss.models.my_system import MySystem

def run_my_system(params: dict, ic: dict, t0: float, t1: float, dt: float):
    system = MySystem(**params)
    x0 = np.array([...], dtype=float)

    T = float(t1 - t0)
    fps = max(1, int(round(1.0 / dt)))

    sol = Solver(system, x0, T=T, fps=fps).run()
    t = sol.t
    X = sol.y.T

    cfg = dict(T=T, dt=dt, params=params, ic=ic)
    out = dict(t=t, X=X)
    return cfg, out
```

## 3) Create a dashboard in `apps/streamlit/components/dashboards/`

Dashboards build a single Plotly figure for the “right panel”.
They typically:
- create a subplot grid,
- plot time series and phase portraits,
- add a cursor / index for animation synchronization if used.

Keep dashboards pure: input is `(cfg, out, controls)`.

## 4) Create a per-system view + `SystemSpec`

Create `apps/streamlit/systems/my_system_view.py` with:

- `controls(prefix)`: Streamlit widgets (parameters, initial conditions, solver settings)
- `run(controls)`: call your runner
- `get_spec()`: return a `SystemSpec`

## 5) Register in the GUI

Add import and entry in `apps/streamlit/registry.py`:

```python
from apps.streamlit.systems.my_system_view import get_spec as my_system_spec

SYSTEM_SPECS["My system"] = my_system_spec()
```

## 6) Sanity checklist

- `dynamics` returns finite values for typical inputs
- No NaNs/Inf after integration (`Solver` will raise)
- `X` orientation matches what the dashboard expects (`(N, n_state)` is recommended)
- Units are consistent (especially for gravity, lengths, motor constants)
