# Streamlit GUI

## Entry point

`apps/streamlit/app.py` is the Streamlit entrypoint:

- loads CSS from `apps/streamlit/assets/style.css` (optional),
- renders a left control column and a right dashboard column,
- uses `apps/streamlit/registry.py` to populate the model selector.

Run:

```bash
streamlit run apps/streamlit/app.py
```

## The `SystemSpec` pattern

Systems are registered as `SystemSpec` objects (see `apps/streamlit/layout.py`).

A `SystemSpec` binds:

- `controls(prefix) -> Controls`: renders Streamlit widgets and returns a dict (must include `run_clicked`)
- `run(controls) -> (cfg, out)`: runs the simulation and returns config + outputs
- `make_dashboard(cfg, out, controls) -> plotly.graph_objects.Figure`: builds a single composed dashboard figure

Fallback mode exists (`make_animation` + `plots[]`) but the preferred UX is the single dashboard.

### Minimal `SystemSpec`

```python
from apps.streamlit.layout import SystemSpec

def controls(prefix: str) -> dict: ...
def run(controls: dict) -> tuple[dict, dict]: ...
def make_dashboard(cfg: dict, out: dict, controls: dict): ...

spec = SystemSpec(
    id="my_system",
    title="My system",
    controls=controls,
    run=run,
    make_dashboard=make_dashboard,
)
```

## Where per-system code lives

For a system named `X`:

- Controls/spec: `apps/streamlit/systems/x_view.py` (`get_spec()`)
- Runner (simulation): `apps/streamlit/runners/x_runner.py`
- Dashboard: `apps/streamlit/components/dashboards/x_dashboard.py`
- Registration: `apps/streamlit/registry.py`

## Output conventions

Runners usually return:

- `cfg`: final numeric settings (T, dt/fps, parameter values)
- `out`: a dict containing at least `t` and `X`:
  - `t`: shape `(N,)`
  - `X`: shape `(N, n_state)` or `(n_state, N)` depending on dashboard (most dashboards expect `(N, n_state)`)

Dashboards typically show:
- time series panels,
- phase portrait(s),
- energy drift or invariants where applicable,
- an animation panel (for mechanical systems).
