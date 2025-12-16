# Core API

## `Solver` (`dss/core/solver.py`)

`Solver` is a light wrapper around `scipy.integrate.solve_ivp`. It:

- builds a time grid (`t_eval`) from `T` and `fps` (unless you pass `t_eval` explicitly),
- calls `system.dynamics(t, x)` on your model,
- checks integration success and guards against NaN/Inf.

### Constructor

```python
Solver(
    system,
    initial_conditions,
    T=10.0,
    fps=60,
    t_span=None,
    t_eval=None,
    method="RK45",
    rtol=1e-6,
    atol=1e-9,
)
```

Notes:
- If `t_span` is not provided, it defaults to `(0.0, T)`.
- `fps` controls the *output sampling* (the density of `t_eval`), not the internal integrator step size.

### Run

```python
sol = Solver(system, x0, T=10.0, fps=200).run()
t = sol.t     # (N,)
X = sol.y.T   # (N, n_state)
```

## Diagnostics helper (`dss/core/experiments.py`)

`run_simulation_with_diagnostics(...)` runs a `Solver`, measures runtime, and returns:

- the `solve_ivp` solution (`sol`),
- a `diagnostics` dict (runtime, number of points, time span, etc.).

It can optionally write a JSONL entry via `SimulationLogger`.

## Logger (`dss/core/logger.py`)

`SimulationLogger` writes one JSON object per run to `logs/runs.jsonl`.

Logged fields typically include:
- model/plant metadata,
- controller metadata,
- wrapper metadata,
- solver configuration,
- initial state,
- runtime / diagnostics,
- optional experiment name and custom metadata.

This is meant to be lightweight and append-only so it works well with quick experiments and plotting scripts.
