# Scripts and artifacts

Offline scripts live in `scripts/`. They are used to:
- generate plots and tables for reports,
- benchmark solver settings,
- validate model behavior independently of Streamlit.

## Running scripts

Run scripts from the repository root (so imports work):

```bash
python scripts/lorenz_attractor.py
```

## Artifact conventions

Scripts should write outputs into `artifacts/`:

- `artifacts/data/` — CSV / NPZ / intermediate data
- `artifacts/figs/` — PNG / PDF figures
- `artifacts/logs/` — run bundles (config + arrays)

`artifacts/` is considered generated output and should not be committed to version control.

## Recommended script structure

A script should follow a consistent shape:

1. Define the model and parameters (or a config dictionary)
2. Run the solver
3. Post-process arrays for plotting
4. Save numeric output and figures into `artifacts/`

Skeleton:

```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dss.models.lorenz import Lorenz
from dss.core.solver import Solver

OUT = Path("artifacts") / "figs"
OUT.mkdir(parents=True, exist_ok=True)

sys = Lorenz()
y0 = np.array([1.0, 1.0, 1.0])

sol = Solver(sys, y0, T=10.0, fps=500, method="DOP853").run()
T = sol.t
X = sol.y.T

plt.figure()
plt.plot(T, X[:, 0])
plt.savefig(OUT / "lorenz_x.png", dpi=200)
```

## Keeping scripts stable

- Avoid importing Streamlit in scripts.
- Keep scripts deterministic where possible (seed any randomness).
- Save enough metadata (config + solver settings) to reproduce figures.
