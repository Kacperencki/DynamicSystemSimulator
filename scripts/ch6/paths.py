"""Common paths for Chapter 6 scripts.

Goal
----
Make Chapter 6 results reproducible and location-independent.

This repository currently contains "legacy" Chapter 6 scripts located in
`scripts/*.py` which write outputs relative to the current working directory
(`data/`, `figs/`, `logs/`). In practice these scripts are typically executed
from within the `scripts/` folder.

The redesign introduces a stable artifact layout under `artifacts/ch6/`.
During migration, `run_all.py` will:
  1) run legacy scripts from `scripts/` (so they keep writing into
     `scripts/{data,figs,logs}`), and
  2) optionally copy/collect the produced files into `artifacts/ch6/`.

Once migration is complete, new scripts should write directly into
`artifacts/ch6/{data,figs,logs}`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Ch6Paths:
    """Resolved key directories."""

    repo_root: Path
    scripts_dir: Path
    legacy_data: Path
    legacy_figs: Path
    legacy_logs: Path

    artifacts_root: Path
    artifacts_data: Path
    artifacts_figs: Path
    artifacts_logs: Path


def resolve_paths() -> Ch6Paths:
    """Resolve repository and artifact directories from this file location."""

    scripts_dir = Path(__file__).resolve().parents[1]  # .../scripts
    repo_root = scripts_dir.parent

    legacy_data = scripts_dir / "data"
    legacy_figs = scripts_dir / "figs"
    legacy_logs = scripts_dir / "logs"

    artifacts_root = repo_root / "artifacts" / "ch6"
    artifacts_data = artifacts_root / "data"
    artifacts_figs = artifacts_root / "figs"
    artifacts_logs = artifacts_root / "logs"

    return Ch6Paths(
        repo_root=repo_root,
        scripts_dir=scripts_dir,
        legacy_data=legacy_data,
        legacy_figs=legacy_figs,
        legacy_logs=legacy_logs,
        artifacts_root=artifacts_root,
        artifacts_data=artifacts_data,
        artifacts_figs=artifacts_figs,
        artifacts_logs=artifacts_logs,
    )
