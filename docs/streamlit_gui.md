# Streamlit GUI

## Entry point

The GUI is launched from the repository root:

```bash
streamlit run streamlit_app.py
```

`streamlit_app.py` is intentionally small:
- sets Streamlit page config (`layout="wide"`),
- loads optional CSS from `apps/streamlit/assets/style.css`,
- renders a two-column layout (controls on the left, content on the right),
- delegates rendering to a selected `SystemSpec`.

## SystemSpec: the GUI “contract”

DSS uses a **SystemSpec** pattern (`apps/streamlit/layout.py`) to keep each system modular.

A `SystemSpec` bundles:

- **Controls**: a function that renders widgets and returns a controls dictionary
- **Run**: a function that turns controls into `(cfg, out)` (numerical result)
- **Dashboard**: a function that turns `(cfg, out)` into a Plotly figure (or multiple panels)

Simplified shape:

```python
Controls = Dict[str, Any]
Cfg = Dict[str, Any]
Out = Dict[str, Any]  # must include at least: T (N,), X (N, n_state)

class SystemSpec:
    controls(prefix: str) -> Controls
    run(controls: Controls) -> tuple[Cfg, Out]
    make_dashboard(cfg: Cfg, out: Out, controls: Controls) -> plotly.graph_objects.Figure
```

## Registry

Available systems are registered in `apps/streamlit/registry.py` as lazy factories:

- `SYSTEM_FACTORIES: Dict[str, Callable[[], SystemSpec]]`
- `SYSTEM_LEGACY: Dict[str, Callable[[], None]]` (temporary, for systems not yet migrated)

Each system lives in `apps/streamlit/systems/<name>_view.py` and must expose:

```python
def get_spec() -> SystemSpec:
    ...
```

## Runners and dashboards

- Runners live in `apps/streamlit/runners/` and are responsible for:
  - translating UI values into model/controller construction,
  - selecting solver settings (`T`, `dt/fps`, method, tolerances),
  - returning `(cfg, out)`.

- Dashboards live in `apps/streamlit/components/` and are responsible for:
  - turning numeric arrays into Plotly figures,
  - optionally creating animations using Plotly frames.

To keep the GUI responsive, runners should:
- avoid generating unnecessarily dense trajectories,
- downsample data used for animation/plotting where appropriate.

## Parameter labels and help tooltips

Every parameter widget in DSS uses a short, readable label (no variable-name clutter) and a `help=` argument that renders as a **?** icon in the UI:

```python
st.number_input(
    "Pole length [m]",
    help="Physical length of the pendulum rod from pivot to tip.",
    ...
)
```

When adding new widgets, follow this convention:
- Keep labels concise (fits in a narrow column without wrapping).
- Put units in the label: `"Mass [kg]"`, `"Speed [rad/s]"`.
- Put the physical explanation in `help=`.

Selectboxes also carry `help=` text explaining each option.

## Session state and reruns

Streamlit reruns the script on every widget interaction. DSS avoids rerunning expensive simulations by:
- storing results in `st.session_state`,
- requiring an explicit **Run** action (usually a button) in the controls.

If you add new widgets, follow the pattern used in existing systems:
- use a system-specific `prefix` for widget keys,
- store “last run” config/output in session state under that prefix.

## Performance and message size

Streamlit sends Plotly figures to the browser as JSON. Large animations can exceed Streamlit’s websocket message size limits.

Typical causes:
- too many time samples (`N` very large),
- too many animation frames,
- each frame repeating the entire history (“trail”) arrays.

Mitigation (recommended):
- limit `fps` for plotting/animation,
- cap number of frames for animated figures,
- avoid full-history traces in every frame.

If you see `MessageSizeError`, reduce the amount of data sent to the browser before increasing Streamlit limits.
