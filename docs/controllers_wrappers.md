# Controllers and wrappers

This project separates **plant dynamics** (models) from **control logic** (controllers) and **composition** (wrappers).

- Controllers compute an input \(u(t, x)\).
- Wrappers combine plant + controller into a single system that can be integrated by the solver.

## Controllers (`dss/controllers/`)

Controllers are primarily used with the inverted pendulum on a cart (cart–pole), but the pattern is reusable.

### LQR (`lqr_controller.py`)

- Linearizes the cart–pole around the upright equilibrium.
- Solves the continuous-time CARE to compute an LQR gain.
- Produces a stabilizing control law near the linearization point.

Typical use:
- only enable LQR when the pendulum is sufficiently close to upright (small angle and/or energy threshold),
- optionally saturate the resulting force.

### Swing-up (`swingup.py`)

- Energy-shaping style control to inject/remove energy until the system approaches the upright region.
- Usually combined with a switching logic to hand over to LQR near upright.

### Switchers (`switcher.py`, `simple_switcher.py`)

- Combine swing-up and stabilizer controllers.
- The “simple” version typically uses a crisp condition (either swing-up or LQR), while the other may include blending or hysteresis.

## Controller interface

A controller should be callable as:

```python
u = controller(t, state)
```

Some controllers also expose explicit methods such as `cart_force(t, state)`; the callable interface is recommended so wrappers can treat controllers uniformly.

## Wrappers (`dss/wrappers/`)

Wrappers are objects that behave like models from the solver’s point of view: they expose a `dynamics(t, state, inputs=None)` method.

### Closed-loop cart wrapper (`closed_loop_cart.py`)

Composes:
- a cart–pole plant model
- a controller that outputs cart force

The wrapper:
1. computes `u = controller(t, state)`
2. calls the plant dynamics with that input

This turns “plant + controller” into a single ODE \(\dot{x} = f(t, x)\) suitable for the solver.

### Motor wrapper (`motor_wrapper.py`)

If you model an actuator separately from the plant (e.g., DC motor producing torque), the wrapper pattern keeps the interface clean:
- the solver integrates one combined state,
- the wrapper routes signals between subsystems.

## Practical guidance

- Keep controllers side-effect free: given `(t, x)`, return `u`.
- Avoid importing Streamlit in controllers/wrappers.
- If you add new control options, add a small unit test ensuring:
  - output is finite,
  - wrapper dynamics returns correct shape,
  - closed-loop runs for a short horizon without NaNs.
