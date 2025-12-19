"""Run the full Chapter 6 experiment pipeline.

This is the first migration step toward a reproducible Chapter 6 build system.

Current behaviour
-----------------
- Uses the Chapter 6 scenario registry (scripts/ch6/scenarios.py)
- Executes the existing legacy scripts in scripts/*.py from within the
  `scripts/` directory (so they continue writing into scripts/{data,figs,logs}).
- Optionally collects produced outputs into artifacts/ch6/{data,figs,logs}.

Usage
-----
From repository root:
    python scripts/ch6/run_all.py --all --collect

Or run specific scenarios:
    python scripts/ch6/run_all.py --only pendulum_small_angle lorenz_attractor

Or filter by tags:
    python scripts/ch6/run_all.py --tag gallery --collect

Notes
-----
This runner intentionally avoids importing the legacy scripts as modules.
Import-time side effects and relative-path outputs are common in scripts;
executing them as subprocesses with cwd=scripts_dir is the most robust.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

# Ensure repository root is on sys.path so `import scripts...` works even when
# this file is executed as: `python scripts/ch6/run_all.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ch6.paths import resolve_paths  # noqa: E402
from scripts.ch6.scenarios import RUN_ORDER, SCENARIOS, ScenarioSpec  # noqa: E402


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)

    sel = p.add_argument_group("selection")
    sel.add_argument("--all", action="store_true", help="Run the default Chapter 6 RUN_ORDER")
    sel.add_argument("--only", nargs="*", default=None, help="Run only these scenario keys")
    sel.add_argument("--tag", action="append", default=None, help="Run scenarios that have this tag (can be repeated)")
    sel.add_argument("--section", action="append", default=None, help="Run scenarios for these thesis sections (e.g. 6.2)")

    run = p.add_argument_group("run control")
    run.add_argument("--collect", action="store_true", help="Copy produced outputs into artifacts/ch6/")
    run.add_argument("--dry-run", action="store_true", help="Print what would be executed, do not run")
    run.add_argument("--keep-going", action="store_true", help="Continue even if a script fails")

    return p.parse_args(argv)


def _select_scenarios(args: argparse.Namespace) -> List[ScenarioSpec]:
    if args.only is not None and len(args.only) > 0:
        keys = args.only
    elif args.all or (args.only is None and args.tag is None and args.section is None):
        keys = RUN_ORDER
    else:
        keys = list(SCENARIOS.keys())

    selected: List[ScenarioSpec] = []
    for k in keys:
        if k not in SCENARIOS:
            raise SystemExit(f"Unknown scenario key: {k!r}. Known: {sorted(SCENARIOS.keys())}")
        selected.append(SCENARIOS[k])

    if args.tag:
        tags = set(args.tag)
        selected = [s for s in selected if tags.intersection(set(s.tags))]

    if args.section:
        sections = set(args.section)
        selected = [s for s in selected if s.section in sections]

    # Preserve RUN_ORDER when possible
    order_index = {k: i for i, k in enumerate(RUN_ORDER)}
    selected.sort(key=lambda s: order_index.get(s.key, 10_000))

    return selected


def _producer_path(scripts_dir: Path, spec: ScenarioSpec) -> Path:
    # legacy: "pendulum_small_ang.py" -> scripts/pendulum_small_ang.py
    # new:    "ch6/convergence_sweeps.py" -> scripts/ch6/convergence_sweeps.py
    return scripts_dir / spec.producer_script


def _run_one(spec: ScenarioSpec, *, scripts_dir: Path, dry_run: bool) -> Tuple[bool, float, str]:
    """Run producer script for a scenario.

    Returns: (ok, seconds, message)
    """
    script_path = _producer_path(scripts_dir, spec)
    if not script_path.exists():
        return False, 0.0, f"Missing producer script: {script_path}"

    cmd = [sys.executable, str(script_path)]
    if dry_run:
        return True, 0.0, f"DRY-RUN: {' '.join(cmd)} (cwd={scripts_dir})"

    env = os.environ.copy()
    # Ensure the repository root is importable for legacy scripts that do `import dss...`
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + py_path if py_path else "")
    # Headless plotting for CI / server environments.
    env.setdefault("MPLBACKEND", "Agg")

    t0 = perf_counter()
    proc = subprocess.run(cmd, cwd=str(scripts_dir), env=env, capture_output=True, text=True)
    dt = perf_counter() - t0

    if proc.returncode == 0:
        msg = proc.stdout.strip() or "OK"
        return True, float(dt), msg

    # failed
    msg = (proc.stdout + "\n" + proc.stderr).strip()
    return False, float(dt), msg


def _collect_outputs(spec: ScenarioSpec, *, scripts_dir: Path, artifacts_root: Path) -> List[str]:
    """Copy outputs listed by a scenario into artifacts/ch6/... (best effort).

    Returns a list of copied artifact relative paths.
    """
    copied: List[str] = []

    for rel in spec.outputs:
        rel_path = Path(rel)

        # If the registry already points into artifacts/, treat as already collected
        # (but still record it in the manifest if the file exists).
        if rel_path.parts and rel_path.parts[0] == "artifacts":
            abs_path = REPO_ROOT / rel_path
            if abs_path.exists() and abs_path.is_file():
                try:
                    copied.append(str(abs_path.relative_to(artifacts_root)))
                except ValueError:
                    # points outside artifacts_root (unexpected); ignore
                    pass
            continue

        src = scripts_dir / rel_path
        if not src.exists():
            continue

        # map data/* -> artifacts/ch6/data/*, figs/* -> artifacts/ch6/figs/*, logs/* -> artifacts/ch6/logs/*
        if rel_path.parts and rel_path.parts[0] in {"data", "figs", "logs"}:
            dst = artifacts_root / rel_path.parts[0] / Path(*rel_path.parts[1:])
        else:
            # unknown, place under artifacts/ch6/misc
            dst = artifacts_root / "misc" / rel_path

        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        copied.append(str(dst.relative_to(artifacts_root)))

    return copied


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])

    paths = resolve_paths()
    # Ensure output roots exist
    paths.legacy_data.mkdir(exist_ok=True)
    paths.legacy_figs.mkdir(exist_ok=True)
    paths.legacy_logs.mkdir(exist_ok=True)
    paths.artifacts_data.mkdir(parents=True, exist_ok=True)
    paths.artifacts_figs.mkdir(parents=True, exist_ok=True)
    paths.artifacts_logs.mkdir(parents=True, exist_ok=True)

    selected = _select_scenarios(args)

    manifest: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "scripts_dir": str(paths.scripts_dir),
        "artifacts_root": str(paths.artifacts_root),
        "selection": [s.key for s in selected],
        "scenarios": {},
    }

    any_fail = False

    for spec in selected:
        ok, seconds, msg = _run_one(spec, scripts_dir=paths.scripts_dir, dry_run=args.dry_run)

        record: Dict[str, object] = {
            "spec": asdict(spec),
            "ok": ok,
            "seconds": seconds,
            "message": msg,
        }

        if args.collect and ok and not args.dry_run:
            record["collected"] = _collect_outputs(
                spec, scripts_dir=paths.scripts_dir, artifacts_root=paths.artifacts_root
            )

        manifest["scenarios"][spec.key] = record

        status = "OK" if ok else "FAIL"
        print(f"[{status}] {spec.key} ({seconds:.3f}s)", flush=True)
        if msg and not args.dry_run:
            # keep this compact; the full output is kept in the manifest
            first_line = msg.splitlines()[0]
            print(f"  {first_line[:200]}", flush=True)

        if not ok:
            any_fail = True
            if not args.keep_going:
                break

    # write manifest (even if a failure happened)
    out_manifest = paths.artifacts_logs / "ch6_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"[DRY-RUN] Manifest would be written to: {out_manifest}", flush=True)
    else:
        print(f"[OK] Wrote manifest: {out_manifest}", flush=True)

    if any_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
