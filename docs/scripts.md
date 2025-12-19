# Offline tools and artifacts

DSS supports running simulations outside the GUI to generate figures/tables for reports (e.g., thesis Chapter 6) and to benchmark solver settings.

## Conventions

- Run tools from the **repository root** so imports work.
- Prefer writing outputs into a dedicated directory under `figures/` (committable) or a temporary `artifacts/` directory (typically not committed).
- Keep runs reproducible: record solver settings (`method`, `rtol`, `atol`, `T`, `fps`) alongside generated outputs.

## Current tools (`tools/`)

### Chapter 6.2 — model “diagnostic cards”

Script:
- `tools/ch6_generate_62_white_blacklines_with_inverted_v2.py`

What it does:
- runs a standard scenario for each model,
- saves grouped PNG “cards” suitable for including in Chapter 6.2.

Example:

```bash
python tools/ch6_generate_62.py \
  --out figures/chapter_05/section6.2 \
  --method DOP853 --rtol 1e-4 --atol 1e-6 --T 10 --fps 200
```

Useful options:
- `--save-csv` additionally writes simple CSV time series (quick debugging)
- `--lorenz-T 50` uses a longer horizon for Lorenz (default 50 s)

### Chapter 6.4 — uniform baseline performance benchmark

Script:
- `tools/ch6_perf_baseline_uniform.py`

What it does:
- runs each model for the same horizon `T` and sampling rate `fps`,
- repeats each case and reports median runtime and median `nfev`,
- writes CSV, LaTeX table, and a summary plot into `--out`.

Example:

```bash
python tools/ch6_perf_baseline_uniform.py \
  --out figures/chapter_05/section6.4 \
  --method DOP853 --rtol 1e-4 --atol 1e-7 --T 10 --fps 200 --repeats 5
```

Useful options:
- `--no-inverted-closed` disables the closed-loop inverted pendulum case.

## Legacy scripts (`scripts_leagacy/`)

The `scripts_leagacy/` folder contains an older, more extensive pipeline (scenario registry, sweep scripts, etc.). It is kept for reference and may diverge from the current API.

If you use it:
- expect to adjust imports and paths,
- treat it as “source material” rather than a supported interface.

## Suggested artifact structure

When generating figures/tables for a thesis chapter, keep outputs grouped:

- `figures/chapter_05/section6.2/` — qualitative behaviour panels
- `figures/chapter_05/section6.4/` — performance tables/plots

This keeps the LaTeX includes stable and makes it clear which scripts produced which files.
