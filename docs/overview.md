# Overview

Dynamic System Simulator (DSS) is organized as a **numerical core** plus an **interactive GUI**:

- **Core library (`dss/`)**: models (ODEs), solver/simulator, controllers, wrappers, and logging utilities.
- **GUI (`apps/streamlit/`)**: Streamlit views and Plotly dashboards that let you configure parameters and run simulations interactively.

The project is designed for *educational use* and *reproducible experiments*: the same model code can be executed from:
- the Streamlit interface,
- offline tools under `tools/`,
- legacy scripts under `scripts_leagacy/` (kept for reference).

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
   - Tools/scripts: save numeric arrays and figures under a chosen output directory (e.g., `figures/...`).
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
  Model definitions (ODEs): pendulums, inverted pendulum, Lorenz, Van der Pol, DC motor.

- `dss/core/`  
  Solver wrapper, high-level simulation API, diagnostics, and logger utilities.

- `dss/controllers/` and `dss/wrappers/`  
  Closed-loop and composition tools (used mainly for the inverted pendulum control case).

- `apps/streamlit/`  
  Streamlit registry, shared UI components, and dashboards.

- `tools/`  
  Supported offline utilities (e.g., figure/table generators for Chapter 6).

- `scripts_leagacy/`  
  Older scripts kept for reference; expect some drift from the current API.
