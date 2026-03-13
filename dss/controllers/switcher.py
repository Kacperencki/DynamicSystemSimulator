# autoswitcher_simple_logged.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import csv
from typing import Any, Optional
from dss.utils.angles import wrap_to_pi


@dataclass
class SimpleTuning:
    theta_full_deg: float = 25.0
    w_full: float = 7.0
    dwell_time: float = 0.07
    min_time_in_mode: float = 0.10

    theta_drop_deg: float = 110.0
    dropout_dwell: float = 0.06

    x_ref_cap: float = 0.12
    x_ref_decay_tau: float = 2.0
    F_min: float = 10.0
    ramp_T: float = 0.15

    # logging
    log_every: float = 0.02   # seconds between telemetry samples (~50 Hz)

class AutoSwitcher:
    """
    Simple Swing-Up → LQR with logging.
    Assumes:
      state = [x, x_dot, theta, theta_dot], theta=0 upright.
      lqr has: K (1x4), x_ref, force_limit, cart_force(t, state).
      swing has: cart_force(t, state).
    """

    def __init__(self, system: Any, lqr_controller: Any, swingup_controller: Any,
                 tuning: SimpleTuning | None = None, verbose: bool = False) -> None:
        self.sys = system
        self.lqr = lqr_controller
        self.swing = swingup_controller
        self.cfg = tuning or SimpleTuning()
        self.verbose = bool(verbose)

        self.th_full = np.deg2rad(self.cfg.theta_full_deg)
        self.th_drop = np.deg2rad(self.cfg.theta_drop_deg)
        self.w_full  = float(self.cfg.w_full)

        self.F_nominal = float(getattr(self.lqr, "force_limit", 30.0))

        self.mode = "SWING"
        self.t_mode = 0.0
        self.armed_since = None
        self.drop_since = None
        self.t0 = None

        self._x_latch = 0.0
        self.lqr.x_ref = 0.0

        # logging
        self.logs = []          # list[dict]
        self._last_telemetry_t = -np.inf

    # ---- logging helpers ----
    def _log_event(self, t: float, kind: str, **extra):
        rec = {"t": float(t), "type": "event", "event": kind, "mode": self.mode}
        rec.update(extra)
        self.logs.append(rec)
        if self.verbose:
            msg = f"[{t:7.3f}] {kind}"
            if extra:
                msg += " " + " ".join(f"{k}={v}" for k, v in extra.items())
            print(msg)

    def _log_telemetry(self, t: float, state, eligible: bool, u_swing: float,
                       u_lqr: float | None, u_out: float, clamped: bool):
        if t - self._last_telemetry_t < self.cfg.log_every:
            return
        self._last_telemetry_t = t
        x, x_dot, th, th_dot = state
        rec = {
            "t": float(t),
            "type": "telemetry",
            "mode": self.mode,
            "x": float(x),
            "x_dot": float(x_dot),
            "theta": float(wrap_to_pi(th)),
            "theta_dot": float(th_dot),
            "eligible": bool(eligible),
            "x_ref": float(getattr(self.lqr, "x_ref", 0.0)),
            "force_limit": float(getattr(self.lqr, "force_limit", self.F_nominal)),
            "u_swing": float(u_swing),
            "u_lqr": float(u_lqr if u_lqr is not None else np.nan),
            "u_out": float(u_out),
            "clamped": bool(clamped),
            "time_in_mode": float(self._time_in_mode(t)),
        }
        self.logs.append(rec)
        if self.verbose:
            print(f"[{t:7.3f}] {rec['mode']:5s} | th={rec['theta']:+6.3f} "
                  f"thd={rec['theta_dot']:+6.3f} elig={int(eligible)} "
                  f"x_ref={rec['x_ref']:+6.3f} F_lim={rec['force_limit']:5.1f} "
                  f"u_sw={rec['u_swing']:+6.2f} u_lqr={rec['u_lqr']:+6.2f} "
                  f"u={rec['u_out']:+6.2f}{' *' if clamped else ''}")

    def get_logs(self, reset: bool = False) -> list[dict]:
        data = self.logs[:] if not reset else self.logs[:]
        if reset:
            self.logs.clear()
        return data

    def dump_csv(self, path: str) -> None:
        if not self.logs:
            return
        # Collect union of keys
        keys = set()
        for r in self.logs: keys |= set(r.keys())
        keys = sorted(keys)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.logs: w.writerow(r)

    # ---- internals ----
    def _time_in_mode(self, t: float) -> float:
        return t - self.t_mode

    def _eligible(self, th_abs: float, thd_abs: float) -> bool:
        return (th_abs <= self.th_full) and (thd_abs <= self.w_full)

    def _enter_lqr(self, t: float, state: np.ndarray) -> None:
        x, x_dot, th, th_dot = state
        th_err = wrap_to_pi(th)
        u_swing = float(self.swing.cart_force(t, state))

        K = np.ravel(np.array(getattr(self.lqr, "K")))
        if K.size != 4:
            raise ValueError("LQR.K must be [kx, kxd, kth, kthd].")
        k1, k2, k3, k4 = map(float, K.tolist())

        eps = 1e-8
        k1_safe = k1 if abs(k1) > eps else (np.sign(u_swing) * eps if u_swing != 0 else eps)
        x_ref = x + (u_swing + k2 * x_dot + k3 * th_err + k4 * th_dot) / k1_safe
        x_ref = float(np.clip(x_ref, -self.cfg.x_ref_cap, self.cfg.x_ref_cap))

        self._x_latch = x_ref
        self.lqr.x_ref = x_ref
        self.lqr.force_limit = float(self.cfg.F_min)

        self.mode = "LQR"
        self.t_mode = t
        self.armed_since = None
        self.drop_since = None

        self._log_event(t, "enter_LQR", x_ref=f"{x_ref:+.3f}", u_swing=f"{u_swing:+.2f}")

    def _update_lqr_decay_and_ramp(self, t: float) -> None:
        elapsed = self._time_in_mode(t)
        tau = max(1e-6, self.cfg.x_ref_decay_tau)
        self.lqr.x_ref = float(np.exp(-elapsed / tau) * self._x_latch)

        if self.cfg.ramp_T > 0.0:
            s = float(np.clip(elapsed / self.cfg.ramp_T, 0.0, 1.0))
            self.lqr.force_limit = float(self.cfg.F_min + (self.F_nominal - self.cfg.F_min) * s)
        else:
            self.lqr.force_limit = self.F_nominal

    # ---- main API ----
    def cart_force(self, t: float, state: np.ndarray) -> float:
        if self.t0 is None:
            self.t0 = t
            self.t_mode = t

        x, x_dot, th, th_dot = state
        th_err = wrap_to_pi(th)
        th_abs = abs(th_err)
        thd_abs = abs(th_dot)

        # --- transitions ---
        if self.mode == "SWING":
            elig = self._eligible(th_abs, thd_abs)
            if elig:
                self.armed_since = t if self.armed_since is None else self.armed_since
                if (t - self.armed_since) >= self.cfg.dwell_time and (t - self.t_mode) >= self.cfg.min_time_in_mode:
                    self._enter_lqr(t, state)
            else:
                if self.armed_since is not None:
                    self._log_event(t, "disarm_capture")
                self.armed_since = None
        else:  # LQR
            if th_abs > self.th_drop and (t - self.t_mode) >= self.cfg.min_time_in_mode:
                self.drop_since = t if self.drop_since is None else self.drop_since
                if (t - self.drop_since) >= self.cfg.dropout_dwell:
                    self.mode = "SWING"
                    self.t_mode = t
                    self.armed_since = None
                    self.drop_since = None
                    self._log_event(t, "drop_to_SWING", theta=f"{th_err:+.3f}")
            else:
                self.drop_since = None

        # --- outputs + telemetry ---
        u_swing = float(self.swing.cart_force(t, state))
        u_lqr = None
        if self.mode == "LQR":
            self._update_lqr_decay_and_ramp(t)
            u_lqr = float(self.lqr.cart_force(t, state))
            F_lim = float(getattr(self.lqr, "force_limit", self.F_nominal))
            clamped = (u_lqr > F_lim) or (u_lqr < -F_lim)
            u_out = float(np.clip(u_lqr, -F_lim, F_lim))
            if clamped:
                self._log_event(t, "saturation", u_lqr=f"{u_lqr:+.2f}", F_lim=f"{F_lim:.1f}")
            self._log_telemetry(t, state, self._eligible(th_abs, thd_abs), u_swing, u_lqr, u_out, clamped)
            return u_out

        # still in swing
        u_out = u_swing
        self._log_telemetry(t, state, self._eligible(th_abs, thd_abs), u_swing, None, u_out, clamped=False)
        return u_out

    # Uniform callable interface: u = pi(t, x)
    def __call__(self, t: float, state: np.ndarray) -> float:
        return self.cart_force(t, state)

