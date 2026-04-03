# apps/streamlit/registry.py
"""
System registry: maps menu display names → SystemSpec factories.

To add a new system to the app
--------------------------------
1. Create apps/streamlit/systems/my_system_view.py with a get_spec() function.
2. Import get_spec here (alias it to avoid name clashes).
3. Add an entry to SYSTEM_FACTORIES with the display name as key.

The factories are called lazily (on first selection) so importing this module
at startup does not construct all specs or import all DSS models upfront.
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict

from apps.streamlit.layout import SystemSpec


SystemFactory = Callable[[], SystemSpec]


def _lazy(module: str, fn: str = "get_spec") -> SystemFactory:
    def _factory() -> SystemSpec:
        mod = importlib.import_module(module)
        return getattr(mod, fn)()
    return _factory


# Lazy factories: each view module is imported only on first selection.
SYSTEM_FACTORIES: Dict[str, SystemFactory] = {
    "Single pendulum": _lazy("apps.streamlit.systems.single_pendulum_view"),
    "Double pendulum": _lazy("apps.streamlit.systems.double_pendulum_view"),
    "Inverted pendulum / cart–pole": _lazy("apps.streamlit.systems.inverted_pendulum_view"),
    "DC motor": _lazy("apps.streamlit.systems.dc_motor_view"),
    "Van der Pol oscillator": _lazy("apps.streamlit.systems.vanderpol_view"),
    "Lorenz system": _lazy("apps.streamlit.systems.lorenz_view"),
}


# Legacy renderers (until migrated to specs).
LegacyFn = Callable[[], None]
SYSTEM_LEGACY: Dict[str, LegacyFn] = {}
