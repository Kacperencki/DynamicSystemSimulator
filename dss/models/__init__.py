# dss/models/__init__.py
"""
Model registry and factory layer.

How to add a new model
-----------------------
1. Create dss/models/my_model.py with a class that has .dynamics(t, state).
2. Import it here.
3. Add a make_my_model() factory function with any needed aliases.
4. Register it in MODEL_REGISTRY under a string key.
5. Use it via: get_model("my_model", mode="...", **kwargs)

Alias dicts (_*_ALIASES)
-------------------------
Map short UI/config parameter names → constructor argument names.
This keeps the UI independent from the model internals.
Example: UI sends "L" → factory translates to "length" before passing to Pendulum().
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict

from .pendulum import Pendulum
from .double_pendulum import DoublePendulum
from .dc_motor import DCMotor
from .vanderpol_circuit import VanDerPol
from .lorenz import Lorenz

# Inverted pendulum is part of DSS, but we expose it here for convenience
from dss.models.inverted_pendulum import InvertedPendulum
from dss.wrappers.motor_wrapper import MotorWrapper


# ---------------------------------------------------------------------
# Helpers: aliasing + signature filtering
# ---------------------------------------------------------------------

def _apply_aliases(kwargs: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Rename short UI/config keys -> constructor keys.
    If destination key already exists, the source is dropped to avoid duplicates.
    """
    out = dict(kwargs)
    for src, dst in mapping.items():
        if src in out and dst not in out:
            out[dst] = out.pop(src)
        else:
            out.pop(src, None)
    return out


def _filter_kwargs(callable_obj: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop keys not accepted by the callable's signature, preventing:
    TypeError: __init__() got an unexpected keyword argument '...'
    """
    sig = inspect.signature(callable_obj)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


# ---------------------------------------------------------------------
# Factories (translation layer between UI params and model constructors)
# ---------------------------------------------------------------------

_PENDULUM_ALIASES = {
    # geometry/params
    "L": "length",
    "m": "mass",
    "g": "gravity",
    # friction
    "b": "damping",
    "fc": "coulomb",
    # drive (if your UI uses these short forms)
    "A": "drive_amplitude",
    "w": "drive_frequency",
    "omega": "drive_frequency",
    "phi": "drive_phase",
}

_DOUBLE_PENDULUM_ALIASES = {
    "L1": "length1",
    "l1": "length1",
    "L2": "length2",
    "l2": "length2",
    "m1": "mass1",
    "m2": "mass2",
    "g": "gravity",
    "b1": "damping1",
    "b2": "damping2",
    "fc1": "coulomb1",
    "fc2": "coulomb2",
    "A1": "drive1_amplitude",
    "w1": "drive1_frequency",
    "omega1": "drive1_frequency",
    "phi1": "drive1_phase",
    "A2": "drive2_amplitude",
    "w2": "drive2_frequency",
    "omega2": "drive2_frequency",
    "phi2": "drive2_phase",
}

_INVERTED_PENDULUM_ALIASES = {
    "L": "length",
    "l": "length",
    "m": "mass",
    "M": "cart_mass",
    "g": "gravity",
    # If your UI uses single friction knobs:
    "b": "b_pend",
    "fc": "coulomb_pend",
    # Optional cart friction short forms (if used)
    "bc": "b_cart",
    "fc_cart": "coulomb_cart",
}

_DC_MOTOR_ALIASES = {
    # common short forms
    "K_e": "Ke",
    "K_t": "Kt",
    "b": "bm",
    "B": "bm",
    # some UIs use "K" for both; map it to Ke if Ke missing
    "K": "Ke",
    # sometimes inertia is called Jm / J
    "Jm": "J",
}


def make_pendulum(mode: str = "default", **kwargs: Any) -> Pendulum:
    kw = _apply_aliases(kwargs, _PENDULUM_ALIASES)
    kw = _filter_kwargs(Pendulum.__init__, kw)
    return Pendulum(mode=mode, **kw)


def make_double_pendulum(mode: str = "default", **kwargs: Any) -> DoublePendulum:
    kw = _apply_aliases(kwargs, _DOUBLE_PENDULUM_ALIASES)
    kw = _filter_kwargs(DoublePendulum.__init__, kw)
    return DoublePendulum(mode=mode, **kw)


def make_inverted_pendulum(mode: str = "default", **kwargs: Any) -> InvertedPendulum:
    kw = _apply_aliases(kwargs, _INVERTED_PENDULUM_ALIASES)
    kw = _filter_kwargs(InvertedPendulum.__init__, kw)
    return InvertedPendulum(mode=mode, **kw)


def make_dc_motor(mode: str = "default", **kwargs: Any) -> DCMotor:
    # DCMotor doesn't use mode, but we accept it for uniform factory signature.
    kw = _apply_aliases(kwargs, _DC_MOTOR_ALIASES)

    # Special case: if "K" was provided but Ke already exists, drop "K"
    if "Ke" in kw and "K" in kwargs:
        kw.pop("Ke", None)  # Ke already correct; ensure no duplicate from aliasing

    kw = _filter_kwargs(DCMotor.__init__, kw)
    return DCMotor(**kw)


def make_motor_wrapper(mode: str = "default", **kwargs: Any) -> MotorWrapper:
    # MotorWrapper doesn't use mode, but we accept it for uniform factory signature.
    kw = _filter_kwargs(MotorWrapper.__init__, dict(kwargs))
    return MotorWrapper(**kw)


def make_vanderpol(mode: str = "default", **kwargs: Any) -> VanDerPol:
    # VanDerPol uses (L, C, mu). Accept mode but ignore.
    kw = _filter_kwargs(VanDerPol.__init__, dict(kwargs))
    return VanDerPol(**kw)


def make_lorenz(mode: str = "default", **kwargs: Any) -> Lorenz:
    # Lorenz uses (sigma, rho, beta). Accept mode but ignore.
    kw = _filter_kwargs(Lorenz.__init__, dict(kwargs))
    return Lorenz(**kw)


# ---------------------------------------------------------------------
# Registry + public factory API
# ---------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "pendulum": make_pendulum,
    "double_pendulum": make_double_pendulum,
    "inverted_pendulum": make_inverted_pendulum,
    "dc_motor": make_dc_motor,
    "motor_wrapper": make_motor_wrapper,
    "vanderpol": make_vanderpol,
    "lorenz": make_lorenz,
}


def get_model(model_name: str, mode: str = "default", **kwargs: Any) -> Any:
    """
    Build a model instance via MODEL_REGISTRY.

    Parameters
    ----------
    model_name:
        Registry key (e.g., "pendulum", "double_pendulum", "lorenz", ...).
    mode:
        Model mode string (used by models that support modes; ignored otherwise).
    kwargs:
        Model parameters (UI/config keys are accepted; factories translate/filter).

    Returns
    -------
    Model instance (or wrapper) that can be simulated.
    """
    try:
        factory = MODEL_REGISTRY[model_name]
    except KeyError as e:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
        ) from e
    return factory(mode=mode, **kwargs)
