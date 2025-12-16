from __future__ import annotations

from typing import Any, Iterable, Mapping

import streamlit as st


def reset_widget_keys(prefix: str, keys: Iterable[str]) -> None:
    """Remove widget state keys so widgets fall back to their default values."""
    for k in keys:
        st.session_state.pop(f"{prefix}_{k}", None)


def clear_run_state(prefix: str) -> None:
    """Remove last simulation results for a system (keeps other app state intact)."""
    st.session_state.pop(f"{prefix}_cfg", None)
    st.session_state.pop(f"{prefix}_out", None)


def apply_preset(prefix: str, values: Mapping[str, Any]) -> None:
    """Apply a preset by writing widget values into st.session_state.

    Keys in `values` are *suffixes* (without the "{prefix}_" part).
    """
    for k, v in dict(values).items():
        st.session_state[f"{prefix}_{k}"] = v


def reset_defaults_button(prefix: str, keys: Iterable[str], *, label: str = "Clear") -> bool:
    """Button that restores startup defaults (by clearing widget keys) and clears outputs."""
    if st.button(label, key=f"{prefix}_reset_defaults", use_container_width=True):
        reset_widget_keys(prefix, keys)
        clear_run_state(prefix)
        st.rerun()
    return False
