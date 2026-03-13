from __future__ import annotations

# controllers/__init__.py

from typing import Any, Type

from dss.controllers.lqr_controller import AutoLQR
from dss.controllers.swingup import AutoSwingUp
from dss.controllers.switcher import AutoSwitcher
from dss.controllers.simple_switcher import SimpleSwitcher

CONTROLLER_REGISTRY: dict[str, Type[Any]] = {
    # Inverted pendulum family
    "ip_lqr": AutoLQR,
    "ip_swingup": AutoSwingUp,
    "ip_switch": AutoSwitcher,
    "ip_switch_simple": SimpleSwitcher,
    # later: "pid_pendulum": PIDController, etc.
}


def get_controller(name: str) -> Type[Any]:
    try:
        return CONTROLLER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown controller '{name}'. "
                         f"Available: {list(CONTROLLER_REGISTRY)}")
