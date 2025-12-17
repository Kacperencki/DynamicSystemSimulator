# Troubleshooting

## Import errors (`ModuleNotFoundError`)

### `No module named 'dss'` or `No module named 'apps'`

Most often:
- you did not install the project in editable mode, or
- you launched Streamlit from the wrong working directory.

Fix:

```bash
python -m pip install -e .
streamlit run streamlit_app.py
```

Run the command from the repository root.

---

## Streamlit `MessageSizeError`

Error looks like:

> Data of size XXX MB exceeds the message size limit ...

Cause: the Plotly figure (especially animations) is too large to serialize and send to the browser.

Fixes (recommended order):
1. Reduce the number of plotted points (lower `fps` or downsample for plotting).
2. Reduce the number of animation frames.
3. Avoid “full history in every frame” trails.

Only if necessary, increase the Streamlit limit in `.streamlit/config.toml`, but this increases memory usage and load time.

---

## Animation runs “faster” than expected

Common causes:
- using a display sampling grid that does not match the intended animation duration,
- frame count is limited and frames are played at a fixed interval,
- Plotly animation settings override real time.

Fix strategy:
- ensure the runner produces a consistent `T` grid,
- cap frames independently from physical simulation time,
- if needed, compute `frame_duration_ms = 1000 * T_total / n_frames`.

---

## Plots look cut off / not full width

This is usually a layout issue (column width). The GUI layout is defined in `streamlit_app.py`:

```python
col_controls, col_main = st.columns([0.8, 4], gap="small")
```

Increase the ratio for the right column if needed.

---

## GUI style does not change after editing CSS

Streamlit can cache resources. If you edit `apps/streamlit/assets/style.css` and do not see changes:
- hard refresh the browser,
- stop and restart Streamlit,
- ensure the CSS path in `streamlit_app.py` points to the file you edited.

---

## Windows path issues in scripts

Prefer `pathlib.Path` and relative paths from the repository root. Avoid hard-coded absolute Windows paths in scripts that you plan to run elsewhere.
