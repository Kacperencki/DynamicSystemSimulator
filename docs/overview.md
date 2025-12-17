# Overview

Dynamic System Simulator (DSS) is organized as a **numerical core** plus an **interactive GUI**:

- **Core library (`dss/`)**: models (ODEs), solver/simulator, controllers, wrappers, and logging utilities.
- **GUI (`apps/streamlit/`)**: Streamlit views and Plotly dashboards that let you configure parameters and run simulations interactively.

The project is designed for *educational use* and *reproducible experiments*: the same model code can be executed from:
- the Streamlit interface,
- offline scripts (`scripts/`),
- unit tests (`tests/`).

## Concepts and data flow

At runtime, DSS follows a simple, repeatable flow:

1. **Controls / configuration**  
   A configuration dictionary is created (either from Streamlit widgets or from a script).

2. **Model construction**  
   The configuration is mapped to a model instance (e.g., `Pendulum`, `Lorenz`, `DCMotor`).

3. **Optional composition**  
   Some systems are used as-is (open-loop). Others are composed via wrappers:
   - plant + controller (closed-loop),
   - actuator/drive wrappers.

4. **Simulation**  
   The solver integrates the ODE and produces a sampled trajectory `(T, X)`.

5. **Outputs**
   - GUI: render Plotly figures and animations.
   - Scripts: save numeric arrays and figures to `artifacts/`.
   - Logger (optional): persist `config.json` + `output.npz` in a run directory.

## Naming and conventions

### Time, states, and shapes

- Time `t` is in **seconds**.
- States are represented as NumPy arrays:
  - single state: `state` shape `(n_state,)`
  - trajectory: `X` shape `(N, n_state)`
  - time grid: `T` shape `(N,)`

### Model interface

Every model provides a continuous-time ODE of the form \(\dot{x} = f(t, x, u)\) via:

- `dynamics(t, state, inputs=None) -> dstate_dt`

The solver calls `dynamics(t, state)` (no inputs). When a model supports external actuation, wrappers/controllers may pass `inputs=...`.

### Solver output vs. solver internal steps

`scipy.integrate.solve_ivp` uses adaptive internal steps. DSS samples the solution on a user-defined output grid (`t_eval`), typically built from:
- simulation duration `T`,
- sampling frequency `fps` (or equivalently `dt ≈ 1/fps`).

This distinction matters because:
- you can request a dense output grid for plotting without forcing the integrator to use tiny internal steps,
- very dense outputs increase payload size in Streamlit.

## Where things live

- `dss/models/`  
  Model definitions (ODEs): pendulums, cart–pole, Lorenz, Van der Pol, DC motor.

- `dss/core/`  
  Solver wrapper, high-level simulation API, diagnostics, and logger utilities.

- `dss/controllers/` and `dss/wrappers/`  
  Closed-loop and composition tools (primarily used for cart–pole control).

- `apps/streamlit/`  
  Streamlit system pages (`systems/`), run logic (`runners/`), and Plotly dashboards (`components/`).

- `scripts/`  
  Offline runs that generate plots/tables; should write outputs to `artifacts/`.

## What “modular” means in DSS

A change is considered “modular” if it does not require editing unrelated layers:

- Adding a new model should not require modifying the solver.
- Adding a new controller should not require modifying models.
- Adding a new Streamlit system should not require changing other systems.

The documentation under **Extending DSS** describes the recommended end-to-end workflow.
