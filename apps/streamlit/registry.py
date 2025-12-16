# apps/streamlit/registry.py

from __future__ import annotations

from typing import Callable, Dict

from apps.streamlit.layout import SystemSpec
from apps.streamlit.systems.single_pendulum_view import get_spec as single_pendulum_spec
from apps.streamlit.systems.double_pendulum_view import get_spec as double_pendulum_spec
from apps.streamlit.systems.vanderpol_view import get_spec as vanderpol_spec
from apps.streamlit.systems.inverted_pendulum_view import get_spec as inverted_pendulum_spec
from apps.streamlit.systems.lorenz_view import get_spec as lorenz_spec
from apps.streamlit.systems.dc_motor_view import get_spec as dc_motor_spec


# Ordered list for the system selector.
SYSTEM_LABELS = [
    "Single pendulum",
    "Double pendulum",
    "Inverted pendulum / cart–pole",
    "DC motor",
    "Van der Pol oscillator",
    "Lorenz system"

]


# Specs that use the new generic renderer.
SYSTEM_SPECS: Dict[str, SystemSpec] = {
    "Single pendulum": single_pendulum_spec(),
    "Double pendulum": double_pendulum_spec(),
    "Inverted pendulum / cart–pole": inverted_pendulum_spec(),
    "DC motor": dc_motor_spec(),
    "Van der Pol oscillator": vanderpol_spec(),
    "Lorenz system": lorenz_spec(),
}


# Legacy renderers (until migrated to specs).
LegacyFn = Callable[[], None]


SYSTEM_LEGACY: Dict[str, LegacyFn] = {}
