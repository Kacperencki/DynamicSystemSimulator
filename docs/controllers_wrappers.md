# Controllers and wrappers

## Controllers (`dss/controllers/`)

These are primarily used with the inverted pendulum (cart–pole), but are written in a reusable way.

- `linearize.py`: linearizes the cart–pole dynamics around the upright equilibrium.
- `lqr_controller.py`: computes an LQR gain from the linearized model (continuous-time CARE).
- `swingup.py`: swing-up logic for bringing the pendulum near the upright position.
- `switcher.py` / `simple_switcher.py`: switch between swing-up and stabilization based on angle/speed thresholds and dwell times.

Typical usage pattern (conceptual):

1. Create a plant (`InvertedPendulum`)
2. Create an LQR controller tuned around the upright
3. Use a swing-up controller to inject energy
4. Switch modes when close enough to upright

The Streamlit runner for the inverted pendulum exposes open-loop and closed-loop runs.

## Wrappers (`dss/wrappers/`)

Wrappers compose dynamics by forwarding states and adding additional states/inputs.

### MotorWrapper (`motor_wrapper.py`)

Adds an electrical state (current) and couples a DC motor to a mechanical plant via gear ratio and efficiency.

State becomes:

- `[i, ...plant_state]`

It computes:
- electrical ODE for `i`,
- motor torque delivered to the plant,
- forwards into `plant.dynamics(t, plant_state, tau_drive)`.

### ClosedLoopCart (`closed_loop_cart.py`)

Wraps a cart–pole plant with a controller object exposing a method like `cart_force(t, state)`.

The wrapper’s `dynamics` calls the controller to compute input force, then calls the plant dynamics.

Note: filename contains a typo (`closed_loop_cart.py`); keep this in mind when importing.
