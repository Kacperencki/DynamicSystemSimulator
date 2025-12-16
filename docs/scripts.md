# Scripts

Scripts in `dss/scripts/` are focused on reproducible runs and figure generation.

## Typical outputs

Many scripts write:

- CSV time series into `dss/scripts/data/`
- PNG figures into `dss/scripts/figs/`

These are suitable for reports/thesis chapters (e.g., consistent “gallery” plots).

## Notable scripts

- `pendulum_small_ang.py` – small-angle pendulum case study
- `double_p_chaos.py` – double pendulum chaotic behavior demo
- `vdp_limit_cycle.py` – Van der Pol limit cycle behavior
- `lorenz_attractor.py` – Lorenz attractor trajectories
- `dc_motor_step.py` – step response of the DC motor model
- `inverted_case_study.py` – inverted pendulum open/closed-loop cases
- `performance.py` – runtime benchmarks and runtime-vs-tolerance sweeps
- `gallery.py` – builds compact “gallery” CSV files from the detailed time series scripts

Run a script from the project root, for example:

```bash
python dss/scripts/performance.py
```
