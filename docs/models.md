# Models

This document describes state definitions, parameters, and common helper methods per model.
For the canonical source of truth, see the class docstrings in `dss/models/*.py`.

## Common model interface

Most models provide:

- `dynamics(t, state, u=None)` → `state_dot`
- `state_labels()` → list of strings for plotting
- `positions(state)` → geometric points for animation (mechanical systems)
- optional checks: `energy_check(...)`, `energy(...)`, etc.

Important: `Solver` calls `system.dynamics(t, state)` with *two* arguments, so models that accept an input must provide a default like `u=None`.

---

## Single pendulum (`dss/models/pendulum.py`)

State:
- `theta` [rad] (absolute from downward vertical)
- `theta_dot` [rad/s]

Constructor (required):
- `length`, `mass`, `mode`

Key parameters (defaults shown):
- `damping=0.0` (viscous)
- `coulomb=0.0` (Coulomb friction)
- `gravity=9.81`
- `drive_amplitude=2.0`, `drive_frequency=2.0`, `drive_phase=0.0`
- `mass_model='point'` (or `'uniform'` via inertia/COM settings)

Helper methods:
- `positions(state)` for animation
- `energy_check(t, state)` for drift diagnostics

Modes:
- `ideal`, `damped`, `driven`, `dc_driven` (see class docstring)

---

## Double pendulum (`dss/models/double_pendulum.py`)

State:
- `theta1`, `theta1_dot`, `theta2`, `theta2_dot`

Constructor (required):
- `length1`, `mass1`, `length2`, `mass2`, `mode`

Key parameters:
- per-joint damping and Coulomb friction: `damping1`, `damping2`, `coulomb1`, `coulomb2`
- optional drives: `drive1_*`, `drive2_*`
- `mass_model='uniform'` by default

Provides:
- `positions(state)` for animation
- energy drift checks and helper solvers

---

## Inverted pendulum on a cart (`dss/models/inverted_pendulum.py`)

State:
- `x`, `x_dot`, `theta`, `theta_dot`

Constructor has defaults; you can instantiate without arguments.

Key parameters:
- pendulum geometry (`length`, `mass`, `lc`, inertias)
- cart mass (`cart_mass`)
- damping and Coulomb friction for cart and pendulum
- drive terms (cart/pendulum excitation)

Modes:
- controlled by `mode` string (default: `'damped_both'`)

Provides:
- `positions(state)` for animation (pivot and tip)
- `energy(state)` and energy drift checks

---

## Van der Pol oscillator (`dss/models/vanderpol_circuit.py`)

Circuit form (parallel LC with nonlinear resistor).

State:
- typically voltage and inductor current (see `state_labels()`)

Parameters:
- `L=1.0`, `C=1.0`, `mu=1.0`

Provides:
- `observables(state)` and labels for additional plotting
- `positions(state)` (used for simple 2D representation if needed)

---

## Lorenz system (`dss/models/lorenz.py`)

State:
- `x`, `y`, `z`

Parameters:
- `sigma=10`, `rho=28`, `beta=8/3` (classic chaotic setting)

Provides:
- `observables(state)` for plotting derived signals

---

## DC motor (`dss/models/dc_motor.py`)

State:
- `i` [A] (armature current)
- `omega` [rad/s] (angular speed)

Equations:
- `L di/dt = v(t) - R i - Ke ω`
- `J dω/dt = Kt i - b ω - τ_load`

Parameters (required):
- `R`, `L`, `Ke`, `Kt`, `Im` (rotor inertia), `bm` (viscous friction)

Inputs:
- `voltage_func`: callable `v(t)` or a constant (default `6.0`)
- `load_func`: callable `tau(t, omega)` or a constant (default `0.0`)
