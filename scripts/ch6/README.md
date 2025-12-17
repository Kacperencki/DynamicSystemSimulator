# Chapter 6 experiment pipeline (WIP)

This folder contains the *new* Chapter 6 build pipeline.

## What is implemented in this step

- `scenarios.py`: scenario registry (single source of truth)
- `run_all.py`: runner that executes scenario producer scripts and writes a manifest
- `paths.py`: consistent path resolution for legacy vs new artifact locations

## Phase 2 (implemented): Chapter 6.2 behaviour panels

- `behaviour_panels.py`: a single producer that generates consistent behaviour
  figures + CSVs for all core models (aligned with Chapter 4 order). It writes
  outputs directly to `artifacts/ch6/...`.

## How to run

From repository root:

Run only the Chapter 6.2 behaviour panels:

```bash
python scripts/ch6/run_all.py --only behaviour_panels --collect
```

Run the default Chapter 6 pipeline:

```bash
python scripts/ch6/run_all.py --all --collect
```

This will:
1. execute the legacy scripts (from `scripts/*.py`) with working directory set to `scripts/`,
2. then copy their listed outputs into `artifacts/ch6/{data,figs,logs}`,
3. write `artifacts/ch6/logs/ch6_manifest.json`.

## Next migration steps (planned)

- Replace legacy scripts with callable experiment functions that accept `ScenarioSpec`.
- Implement `convergence_sweeps` and `gui_timings` producers under this folder.
