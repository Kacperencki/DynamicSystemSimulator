from __future__ import annotations

# dss/controllers/simple_switcher.py
"""
Hard-switching supervisor: swing-up ↔ LQR.

State machine
-------------
   SWING ──────────────────────────► LQR
           all engage conditions met
                                     │
                                     │  any exit condition exceeded
                                     ▼
   SWING ◄──────────────────────────
           (if allow_dropout=True)

Engage conditions (all must hold simultaneously):
  |θ| < engage_angle_deg  AND  |θ̇| < engage_speed_rad_s  AND  |ẋ| < engage_cart_speed

Exit conditions (any one triggers dropout):
  |θ| > dropout_angle_deg  OR  |θ̇| > dropout_speed_rad_s  OR  |ẋ| > dropout_cart_speed

On ENTER LQR: sets lqr.x_ref = current cart position to prevent a sudden
              cart-centering snap that could destabilise the pole.

Note: blend_time and du_max are accepted for UI/runner compatibility but
      unused here — this is a hard (instantaneous) switch with no blending.
"""

import numpy as np
from typing import Any, Optional

from dss.utils.angles import wrap_to_pi


class SimpleSwitcher:
    """
    Hard swing-up ↔ LQR switcher (NO blending).

    - Output is strictly from the active controller (SWING or LQR).
    - On ENTER LQR: set LQR.x_ref = current x (prevents cart snapping back to 0).
    - Optional dropout hysteresis is kept.

    NOTE: we keep blend_time/du_max in signature for compatibility with your runner/UI,
    but they are intentionally ignored here.
    """

    SWING = "SWING"
    LQR = "LQR"

    def __init__(
        self,
        system: Any,
        lqr_controller: Any,
        swingup_controller: Any,
        engage_angle_deg: float = 25.0,
        engage_speed_rad_s: float = 9.0,
        engage_cart_speed: float = 1.2,
        hold_time: float = 0.05,             # compat, unused
        dropout_angle_deg: float = 45.0,
        dropout_speed_rad_s: float = 30.0,
        dropout_cart_speed: float = 10.0,
        allow_dropout: bool = True,
        min_time_in_mode: float = 0.10,      # compat, unused
        verbose: bool = False,
        # kept for compat but unused (hard switch)
        blend_time: float = 0.0,
        du_max: Optional[float] = None,
        # optional safety: if not None, force SWING when |theta| exceeds this
        lqr_failsafe_angle_deg: Optional[float] = 120.0,
    ) -> None:
        self.sys = system
        self.c_lqr = lqr_controller
        self.c_swing = swingup_controller

        # Entry thresholds (into LQR)
        self.th_on = np.deg2rad(float(engage_angle_deg))
        self.w_on = float(engage_speed_rad_s)
        self.xd_on = float(engage_cart_speed)

        # Exit thresholds (out of LQR)
        self.th_off = np.deg2rad(float(dropout_angle_deg))
        self.w_off = float(dropout_speed_rad_s)
        self.xd_off = float(dropout_cart_speed)
        self.allow_dropout = bool(allow_dropout)

        self.verbose = bool(verbose)
        self.mode = self.SWING

        self._last_log_t = -1.0

        self._lqr_failsafe = None if lqr_failsafe_angle_deg is None else np.deg2rad(float(lqr_failsafe_angle_deg))

    def _log(self, msg: str, t: float, every: float = 0.05) -> None:
        if not self.verbose:
            return
        if self._last_log_t < 0.0 or (t - self._last_log_t) >= every:
            self._last_log_t = float(t)
            print(msg)

    def _should_enter_lqr(self, theta_w: float, theta_dot: float, x_dot: float) -> bool:
        return (abs(theta_w) < self.th_on) and (abs(theta_dot) < self.w_on) and (abs(x_dot) < self.xd_on)

    def _should_exit_lqr(self, theta_w: float, theta_dot: float, x_dot: float) -> bool:
        if not self.allow_dropout:
            return False
        return (abs(theta_w) > self.th_off) or (abs(theta_dot) > self.w_off) or (abs(x_dot) > self.xd_off)

    @staticmethod
    def _clip(u: float, limit: Optional[float]) -> float:
        if limit is None:
            return float(u)
        lim = float(abs(limit))
        return float(np.clip(u, -lim, lim))

    def cart_force(self, t: float, state: np.ndarray) -> float:
        x, x_dot, theta, theta_dot = state
        t = float(t)

        theta_w = wrap_to_pi(float(theta))

        # Optional safety: if LQR is asked to run too far away, force swing-up.
        if self._lqr_failsafe is not None and abs(theta_w) > self._lqr_failsafe:
            self.mode = self.SWING

        # Mode transitions
        if self.mode == self.SWING:
            if self._should_enter_lqr(theta_w, float(theta_dot), float(x_dot)):
                # Critical for "stable cart": hold current cart position as reference
                if hasattr(self.c_lqr, "x_ref"):
                    self.c_lqr.x_ref = float(x)
                self.mode = self.LQR
                self._log(f"[switch] ENTER LQR @ t={t:.3f}, x_ref={float(x):+.3f}", t)
        else:
            if self._should_exit_lqr(theta_w, float(theta_dot), float(x_dot)):
                self.mode = self.SWING
                self._log(f"[switch] EXIT  LQR @ t={t:.3f}", t)

        # Hard selection + per-controller saturation
        if self.mode == self.LQR:
            u = float(self.c_lqr.cart_force(t, state))
            u = self._clip(u, getattr(self.c_lqr, "force_limit", None))
            return u

        u = float(self.c_swing.cart_force(t, state))
        swing_limit = getattr(self.c_swing, "F_max", getattr(self.c_swing, "force_limit", None))
        u = self._clip(u, swing_limit)
        return u

    # Uniform callable interface: u = pi(t, x)
    def __call__(self, t: float, state: np.ndarray) -> float:
        return self.cart_force(t, state)
