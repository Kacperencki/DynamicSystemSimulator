# Extending DSS

This page describes the recommended workflow for adding a new dynamical system end-to-end:

1. Add a model under `dss/models/`
2. Register it (so it can be created by name)
3. (Optional) add controllers/wrappers
4. Add a Streamlit SystemSpec under `apps/streamlit/systems/`
5. Register the system in `apps/streamlit/registry.py`

The goal is that **each new system looks like the existing ones**: same structure, same run flow, minimal copy-paste.

---

## 1) Add a new model

Create a new file:

```
dss/models/my_system.py
```

Minimum required API:

```python
import numpy as np

class MySystem:
    def __init__(self, ...):
        ...

    def state_labels(self):
        return ["x1", "x2", ...]

    def dynamics(self, t, state, inputs=None):
        # return dx/dt as a 1D array of length n_state
        return np.array([...], dtype=float)
```

Optional but recommended:
- `positions(state)` for animations
- `energy_check(state)` for validation plots

---

## 2) Register the model

Add it to `dss/models/__init__.py` so it can be created by name through the registry. The file contains two alias dicts — add an entry to both:

```python
# in dss/models/__init__.py
from dss.models.my_system import MySystem

# Human-readable display name → class
MODEL_CLASSES["My System"] = MySystem

# Short key used in config dicts
MODEL_ALIASES["my_system"] = MySystem
```

After this, `build_system(cfg)` in `dss/core/pipeline.py` can construct your model from a plain config dict.

---

## 3) Optional: add controllers and wrappers

Controllers live in `dss/controllers/` and should be callable:

```python
u = controller(t, state)
```

Wrappers live in `dss/wrappers/` and expose `dynamics(t, state, inputs=None)` so they can be integrated by the solver like a normal model.

---

## 4) Add a Streamlit SystemSpec

Create:

```
apps/streamlit/systems/my_system_view.py
```

Follow the pattern used in existing views:

- render a small set of parameter widgets (use `help=` on every widget — see `docs/streamlit_gui.md`),
- call a runner function that returns `(cfg, out)`,
- render a Plotly dashboard.

Skeleton:

```python
from __future__ import annotations
from typing import Any, Dict

import streamlit as st
from apps.streamlit.layout import SystemSpec
from apps.streamlit.runners.my_system_runner import run_my_system
from apps.streamlit.components.dashboards.my_system_dashboard import make_my_system_dashboard

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]

def controls(prefix: str) -> Controls:
    # Widgets must use the prefix for unique keys
    p = lambda k: f"{prefix}_{k}"
    run_clicked = st.button("Run", key=p("run"))
    # add widgets...
    return {"run_clicked": run_clicked, ...}

def get_spec() -> SystemSpec:
    return SystemSpec(
        title="My system",
        controls=controls,
        run=run_my_system,
        make_dashboard=make_my_system_dashboard,
    )
```

---

## 5) Add a runner

Create:

```
apps/streamlit/runners/my_system_runner.py
```

Runners should:
- build the model from control values,
- set up the solver settings,
- return a JSON-serializable config dict plus an output dict with at least `T` and `X`.

---

## 6) Add a dashboard

Create:

```
apps/streamlit/components/dashboards/my_system_dashboard.py
```

Keep dashboards pure and deterministic:
- accept `cfg`, `out`, `controls`
- return a Plotly `Figure`

---

## 7) Register the system in the GUI registry

Update `apps/streamlit/registry.py`:

```python
from apps.streamlit.systems.my_system_view import get_spec as my_system_spec

SYSTEM_FACTORIES["My system"] = my_system_spec
```

---

## Checklist

Before considering the system “done”:

- [ ] model runs via `dss.core.solver.Solver`
- [ ] Streamlit system loads without import errors
- [ ] runner returns `out["T"]` and `out["X"]` with correct shapes
- [ ] plots render for at least one preset
- [ ] (optional) add a quick `pytest` smoke test
