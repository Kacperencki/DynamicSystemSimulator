from __future__ import annotations

from typing import Iterable
import streamlit as st


def reset_widget_keys(prefix: str, keys: Iterable[str]) -> None:
    """Remove widget state keys so widgets fall back to their default values."""
    for k in keys:
        st.session_state.pop(f"{prefix}_{k}", None)


def reset_defaults_button(prefix: str, keys: Iterable[str], *, label: str = "Reset defaults") -> bool:
    """Button that resets widget state for a given prefix, then reruns."""
    if st.button(label, key=f"{prefix}_reset_defaults", width='stretch'):
        reset_widget_keys(prefix, keys)
        st.rerun()
    return False
