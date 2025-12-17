# Models

This page documents the built-in dynamical systems under `dss/models/`.

## Common conventions

- Time `t` is in **seconds**.
- Angles are in **radians**.
- State vectors are NumPy arrays:
  - single state `state`: shape `(n_state,)`
  - trajectory `X`: shape `(N, n_state)`
- All models provide:
  - `dynamics(t, state, inputs=None) -> dstate_dt`
  - `state_labels() -> list[str]`
- Some models additionally provide:
  - `positions(state)` for animation geometry
  - `energy_check(state)` for energy partition / validation

---

## Single pendulum (`Pendulum`)

**File:** `dss/models/pendulum.py`  
**State:** `x = [theta, theta_dot]`

- `theta`: angle from **downward** vertical (rad)
- `theta_dot`: angular velocity (rad/s)

### Modes

The `mode` string controls which terms are active:

- `ideal`: gravity only
- `damped`: gravity + viscous damping + Coulomb friction
- `driven`: damped + harmonic torque `A*cos(ω t + φ)`
- `dc_driven`: damped + external torque passed via `inputs` (also referred to as `tau_drive` conceptually)

### Key parameters

- `length`, `mass`
- `gravity` (default 9.81)
- `damping`: viscous coefficient
- `coulomb`: Coulomb friction level
- `drive_amplitude`, `drive_frequency`, `drive_phase`
- `mass_model`: `"point"` or `"uniform"` (controls inertia/com geometry)
- `I`, `lc`: optional overrides (inertia and COM distance)
- `coulomb_vel_eps`: smoothing for Coulomb friction around zero velocity

### Geometry and energy

- `positions(state)` returns key points for animation (pivot/COM/tip).
- `energy_check(state)` returns energy parts used for validation plots.

---

## Double pendulum (`DoublePendulum`)

**File:** `dss/models/double_pendulum.py`  
**State:** `x = [theta1, theta1_dot, theta2, theta2_dot]`

Angles are absolute (each link measured from downward vertical). The second link angle is not relative by default.

### Modes

- `ideal`
- `damped`
- `driven` (harmonic torques on joint 1 and/or 2)
- `dc_driven` (external torques passed via `inputs` conceptually)

### Key parameters

- `length1, mass1`, `length2, mass2`
- `damping1, damping2` and `coulomb1, coulomb2`
- `drive1_*` and `drive2_*` (amplitude/frequency/phase per joint)
- `mass_model`: `"uniform"` or `"point"`
- `I1, lc1, I2, lc2`: optional overrides
- `coulomb_vel_eps`: smoothing around zero velocity

### Geometry and energy

- `positions(state)` returns pivot, mass1, mass2 coordinates.
- `energy_check(state)` returns energy parts suitable for conservation/drift checks.

---

## Inverted pendulum on a cart (`InvertedPendulum`)

**File:** `dss/models/inverted_pendulum.py`  
**State:** `x = [cart_pos, cart_vel, theta, theta_dot]`

- `cart_pos`: cart position along track (m)
- `cart_vel`: cart velocity (m/s)
- `theta`: pendulum angle (rad), measured from downward vertical
- `theta_dot`: angular velocity (rad/s)

### Modes

The cart–pole model supports viscous + Coulomb friction and optional harmonic drives:

- friction:
  - `b_cart`, `coulomb_cart`
  - `b_pend`, `coulomb_pend`
- drives:
  - `cart_drive_*`: external force on cart
  - `pend_drive_*`: external torque on pivot

Coulomb friction is smoothed using `coulomb_k` (gain for `tanh` smoothing).

### Key parameters

- `length`, `mass` (pendulum), `cart_mass`
- `mass_model`: `"point"` or `"uniform"` (inertia and COM location)
- `I_com`, `lc`: optional overrides
- `gravity`

### Geometry and energy

- `positions(state)` returns cart pivot and pendulum tip coordinates.
- `energy_check(state)` is provided for diagnostic plots.

---

## Lorenz system (`Lorenz`)

**File:** `dss/models/lorenz.py`  
**State:** `x = [x, y, z]`

The Lorenz system is an autonomous ODE and does not use external inputs.

### Parameters

- `sigma`, `rho`, `beta`

### Notes

- Lorenz is commonly simulated on shorter horizons (e.g., 5–30 seconds) with moderate tolerances.
- Visualization typically uses:
  - 3D trajectory `(x(t), y(t), z(t))`,
  - time series for each coordinate,
  - phase-plane projections.

---

## Van der Pol oscillator (circuit form) (`VanDerPol`)

**File:** `dss/models/vanderpol_circuit.py`  
**State:** `x = [v, iL]`

- `v`: capacitor voltage (V)
- `iL`: inductor current (A)

The model implements a parallel LC with a nonlinear current source.

### Parameters

- `L`: inductance
- `C`: capacitance
- `mu`: nonlinearity strength in `i_nl(v) = mu*(v^3/3 - v)`

### Notes

- The GUI often displays phase portraits (`v` vs `iL`) and time series.
- This model does not define `energy_check()` by default (you can add one if needed).

---

## DC motor (`DCMotor`)

**File:** `dss/models/dc_motor.py`  
**State:** `x = [i, omega]`

- `i`: armature current (A)
- `omega`: angular speed (rad/s)

### Parameters

Electrical:
- `R`, `L`, `Ke`

Mechanical:
- `Kt`, `J` (or alias `Im`), `bm`

### Inputs

The motor includes built-in “runner-style” voltage/load configuration options used by the GUI:
- voltage modes: constant, step, sine, PWM-like options (see file docstring)
- load modes: none, constant, viscous, Coulomb-like (see file docstring)

### Energy and diagnostics

`energy_check(state)` is available and can be used to validate numeric behavior under different loads/voltages.
