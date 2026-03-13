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


def reset_to_preset(
    prefix: str,
    keys: Iterable[str],
    preset_values: Mapping[str, Any],
    *,
    preset_name: str | None = None,
    preset_key_suffix: str = "preset",
) -> None:
    """Reset a set of widget keys to a named preset.

    For every suffix in `keys`:
      - if present in `preset_values` -> write that value
      - otherwise -> remove the key so the widget falls back to its declared default

    If `preset_name` is provided, the preset selector (key f"{prefix}_{preset_key_suffix}")
    is set accordingly.

    IMPORTANT (Streamlit): Call this from a widget callback (on_click / on_change) when
    any of the affected widgets already exist on the page, otherwise Streamlit may raise
    StreamlitAPIException (cannot modify widget state after instantiation).
    """
    # First, restore the preset selector to a consistent state.
    if preset_name is not None:
        st.session_state[f"{prefix}_{preset_key_suffix}"] = preset_name

    # Then reset widget values.
    for k in keys:
        full = f"{prefix}_{k}"
        if k in preset_values:
            st.session_state[full] = preset_values[k]
        else:
            st.session_state.pop(full, None)


def make_reset_callback(
    prefix: str,
    keys: Iterable[str],
    preset_values: Mapping[str, Any] | None = None,
    preset_name: str | None = None,
    preset_key_suffix: str = "preset",
):
    """Return a zero-argument callback that resets widget keys and clears run state.

    Shared by reset_defaults_button() and run_clear_row_form() so the logic lives
    in exactly one place.
    """
    _keys = list(keys)

    def _callback() -> None:
        if preset_values is not None:
            reset_to_preset(
                prefix,
                _keys,
                preset_values,
                preset_name=preset_name,
                preset_key_suffix=preset_key_suffix,
            )
        else:
            reset_widget_keys(prefix, _keys)
        clear_run_state(prefix)

    return _callback


def reset_defaults_button(
    prefix: str,
    keys: Iterable[str],
    *,
    label: str = "Clear",
    preset_values: Mapping[str, Any] | None = None,
    preset_name: str | None = None,
    preset_key_suffix: str = "preset",
) -> bool:
    """Button that restores defaults and clears outputs.

    If `preset_values` is provided, restores that preset.
    Otherwise, clears widget keys so widgets fall back to their declared defaults.
    """
    st.button(
        label,
        key=f"{prefix}_reset_defaults",
        use_container_width=True,
        on_click=make_reset_callback(prefix, keys, preset_values, preset_name, preset_key_suffix),
    )
    return False
