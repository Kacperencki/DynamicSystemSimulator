from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import streamlit as st

from apps.streamlit.components.controls_common import (
    apply_preset,
    clear_run_state,
    make_reset_callback,
    reset_defaults_button,
    reset_to_preset,
    reset_widget_keys,
)


@dataclass(frozen=True)
class SliderSpec:
    label: str
    min: int
    max: int
    value: int
    step: int


def run_reset_row(prefix: str, reset_keys: Iterable[str], *, reset_label: str = "Clear") -> bool:
    """Legacy (non-form) row: Run + Clear."""
    c1, c2 = st.columns([1, 1])
    with c1:
        run_clicked = st.button("Run", key=f"{prefix}_run", type="primary", use_container_width=True)
    with c2:
        reset_defaults_button(prefix, list(reset_keys), label=reset_label)
    return bool(run_clicked)


def run_clear_row_form(
    prefix: str,
    reset_keys: Iterable[str],
    *,
    clear_label: str = "Clear",
    default_preset: Mapping[str, Any] | None = None,
    default_preset_name: str | None = None,
    preset_key_suffix: str = "preset",
) -> bool:
    """Form-friendly row: Run + Clear.

    Must be called inside `with st.form(...):`.

    If `default_preset` is provided, the Clear button restores that preset (Option B).
    Otherwise, it clears widget keys so widgets fall back to their declared defaults.

    IMPORTANT (Streamlit): Clear action is implemented via `on_click` callback, so we can
    safely write into st.session_state for already-instantiated widgets.
    """

    c1, c2 = st.columns([1, 1])
    with c1:
        run_clicked = st.form_submit_button("Run", type="primary", use_container_width=True)
    with c2:
        st.form_submit_button(
            clear_label,
            use_container_width=True,
            on_click=make_reset_callback(prefix, reset_keys, default_preset, default_preset_name, preset_key_suffix),
        )

    return bool(run_clicked)


def presets_selector(
    prefix: str,
    presets: Mapping[str, Mapping[str, Any]],
    *,
    label: str = "Preset",
    key_suffix: str = "preset",
    default_name: str | None = "Default",
    clear_outputs: bool = True,
) -> str:
    """Select a built-in preset and apply it to widget state.

    This widget should be placed *before* other widgets that it affects.
    """
    names = list(presets.keys())
    if not names:
        return ""

    key = f"{prefix}_{key_suffix}"
    default_idx = names.index(default_name) if default_name in names else 0

    def _on_change() -> None:
        name = str(st.session_state.get(key, names[default_idx]))
        if name in presets:
            apply_preset(prefix, presets[name])
            if clear_outputs:
                clear_run_state(prefix)

    st.selectbox(label, names, index=default_idx, key=key, on_change=_on_change)
    return str(st.session_state.get(key, names[default_idx]))


def solver_settings(
    prefix: str,
    *,
    expanded: bool = False,
    methods: tuple[str, str, str] = ("RK45", "Radau", "DOP853"),
    method_default: str = "RK45",
    rtol_default: float = 1e-4,
    atol_default: float = 1e-7,
) -> Dict[str, Any]:
    """Solver selection + tolerances."""
    with st.expander("Numerical solver", expanded=expanded):
        # Full-width method selector, then tolerances below.
        opts = list(methods)
        idx = opts.index(method_default) if method_default in opts else 0
        st.selectbox(
            "Method",
            opts,
            index=idx,
            key=f"{prefix}_solver_method",
            help="ODE solver used for numerical integration.",
        )

        c1, c2 = st.columns(2)
        with c1:
            st.number_input(
                "rtol",
                value=float(rtol_default),
                min_value=1e-12,
                max_value=1.0,
                format="%.1e",
                key=f"{prefix}_rtol",
                help="Relative tolerance (smaller = higher accuracy, slower).",
            )
        with c2:
            st.number_input(
                "atol",
                value=float(atol_default),
                min_value=1e-12,
                max_value=1.0,
                format="%.1e",
                key=f"{prefix}_atol",
                help="Absolute tolerance (smaller = higher accuracy, slower).",
            )

    return dict(
        solver_method=str(st.session_state.get(f"{prefix}_solver_method", method_default)),
        rtol=float(st.session_state.get(f"{prefix}_rtol", rtol_default)),
        atol=float(st.session_state.get(f"{prefix}_atol", atol_default)),
    )


def simulation_time(
    prefix: str,
    *,
    expanded: bool = False,
    t0_default: float = 0.0,
    t1_default: float = 10.0,
    dt_default: float = 0.01,
    t0_min: Optional[float] = None,
    t0_max: Optional[float] = None,
    t1_min: Optional[float] = None,
    t1_max: Optional[float] = None,
    dt_min: float = 1e-5,
    dt_step: float = 0.001,
    dt_format: str = "%.6f",
) -> Tuple[float, float, float]:
    with st.expander("Simulation time", expanded=expanded):
        c1, c2, c3 = st.columns(3)
        with c1:
            kwargs = dict(value=t0_default, key=f"{prefix}_t0")
            if t0_min is not None:
                kwargs["min_value"] = float(t0_min)
            if t0_max is not None:
                kwargs["max_value"] = float(t0_max)
            st.number_input("t₀ [s]", help="Simulation start time.", **kwargs)

        with c2:
            kwargs = dict(value=t1_default, key=f"{prefix}_t1")
            if t1_min is not None:
                kwargs["min_value"] = float(t1_min)
            if t1_max is not None:
                kwargs["max_value"] = float(t1_max)
            st.number_input("t₁ [s]", help="Simulation end time.", **kwargs)

        with c3:
            st.number_input(
                "Δt [s]",
                value=dt_default,
                min_value=float(dt_min),
                step=float(dt_step),
                format=dt_format,
                key=f"{prefix}_dt",
                help="Sampling time for stored results and animation.",
            )

    t0 = float(st.session_state.get(f"{prefix}_t0", t0_default))
    t1 = float(st.session_state.get(f"{prefix}_t1", t1_default))
    dt = float(st.session_state.get(f"{prefix}_dt", dt_default))
    return t0, t1, dt


def animation_performance(
    prefix: str,
    *,
    title: str = "Animation / performance",
    expanded: bool = False,
    layout: str = "single",  # "single" | "two_columns"
    fps: SliderSpec,
    max_frames: SliderSpec,
    max_plot_pts: SliderSpec,
    trail_default: bool,
    trail_checkbox_label: str = "Show tip trail",
    trail_max_points: SliderSpec,
) -> dict:
    with st.expander(title, expanded=expanded):
        if layout == "two_columns":
            c1, c2 = st.columns(2)
            with c1:
                st.slider(
                    fps.label,
                    fps.min,
                    fps.max,
                    fps.value,
                    fps.step,
                    key=f"{prefix}_fps_anim",
                    help="Target animation frame rate.",
                )
                st.slider(
                    max_frames.label,
                    max_frames.min,
                    max_frames.max,
                    max_frames.value,
                    max_frames.step,
                    key=f"{prefix}_max_frames",
                    help="Maximum number of animation frames rendered.",
                )
                st.slider(
                    max_plot_pts.label,
                    max_plot_pts.min,
                    max_plot_pts.max,
                    max_plot_pts.value,
                    max_plot_pts.step,
                    key=f"{prefix}_max_plot_pts",
                    help="Maximum number of points drawn in plots (decimation for speed).",
                )
            with c2:
                st.checkbox(
                    trail_checkbox_label,
                    value=trail_default,
                    key=f"{prefix}_trail_on",
                    help="Show a short history (trail) instead of the full curve.",
                )
                st.session_state.setdefault(f"{prefix}_live_plots", False)
                st.checkbox(
                    "Live plots",
                    key=f"{prefix}_live_plots",
                    help=(
                        "When enabled, the right-side time-series and phase plots animate "
                        "frame-by-frame alongside the main animation. This looks great but "
                        "generates a very large figure payload (10-50 MB) which can cause "
                        "slow loading and sluggish browser performance. "
                        "Disable for significantly faster rendering."
                    ),
                )
                st.slider(
                    trail_max_points.label,
                    trail_max_points.min,
                    trail_max_points.max,
                    trail_max_points.value,
                    trail_max_points.step,
                    key=f"{prefix}_trail_max_points",
                    help="Number of points kept in the trail.",
                )
        else:
            st.slider(
                fps.label,
                fps.min,
                fps.max,
                fps.value,
                fps.step,
                key=f"{prefix}_fps_anim",
                help="Target animation frame rate.",
            )
            st.slider(
                max_frames.label,
                max_frames.min,
                max_frames.max,
                max_frames.value,
                max_frames.step,
                key=f"{prefix}_max_frames",
                help="Maximum number of animation frames rendered.",
            )

            st.checkbox(
                trail_checkbox_label,
                value=trail_default,
                key=f"{prefix}_trail_on",
                help="Show a short history (trail) instead of the full curve.",
            )
            st.session_state.setdefault(f"{prefix}_live_plots", False)
            st.checkbox(
                "Live plots",
                key=f"{prefix}_live_plots",
                help=(
                    "When enabled, the right-side time-series and phase plots animate "
                    "frame-by-frame alongside the main animation. This looks great but "
                    "generates a very large figure payload (10-50 MB) which can cause "
                    "slow loading and sluggish browser performance. "
                    "Disable for significantly faster rendering."
                ),
            )
            st.slider(
                trail_max_points.label,
                trail_max_points.min,
                trail_max_points.max,
                trail_max_points.value,
                trail_max_points.step,
                key=f"{prefix}_trail_max_points",
                help="Number of points kept in the trail.",
            )

            st.slider(
                max_plot_pts.label,
                max_plot_pts.min,
                max_plot_pts.max,
                max_plot_pts.value,
                max_plot_pts.step,
                key=f"{prefix}_max_plot_pts",
                help="Maximum number of points drawn in plots (decimation for speed).",
            )

    return dict(
        fps_anim=int(st.session_state.get(f"{prefix}_fps_anim", fps.value)),
        max_frames=int(st.session_state.get(f"{prefix}_max_frames", max_frames.value)),
        trail_on=bool(st.session_state.get(f"{prefix}_trail_on", trail_default)),
        trail_max_points=int(st.session_state.get(f"{prefix}_trail_max_points", trail_max_points.value)),
        max_plot_pts=int(st.session_state.get(f"{prefix}_max_plot_pts", max_plot_pts.value)),
        live_plots=bool(st.session_state.get(f"{prefix}_live_plots", False)),
    )


def logging_settings(
    prefix: str,
    *,
    expanded: bool = False,
    default_dir: str = "logs",
) -> tuple[bool, str, str]:
    """Optional persistence of runs (config + arrays)."""
    with st.expander("Logging", expanded=expanded):
        save_run = st.checkbox(
            "Save run (config + output)",
            value=False,
            key=f"{prefix}_save_run",
            help="Save configuration and simulation outputs to disk.",
        )
        log_dir = st.text_input(
            "Log directory",
            value=str(default_dir),
            key=f"{prefix}_log_dir",
            help="Folder where run artifacts will be saved.",
        )
        run_name = st.text_input(
            "Run name (optional)",
            value="",
            key=f"{prefix}_run_name",
            help="Optional name appended to the saved files.",
        )
    return bool(save_run), str(log_dir), str(run_name)
