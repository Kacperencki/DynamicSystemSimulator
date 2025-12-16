import numpy as np
from typing import Optional


def wrap_to_pi(th: float) -> float:
    return ((th + np.pi) % (2 * np.pi)) - np.pi


class SimpleSwitcher:
    """Simple swing-up ↔ LQR switcher.

    Fixes two practical issues that make swing-up look "jerky":
      1) It clips by the *active* controller limit (swing-up vs LQR),
         instead of always clipping by the LQR limit.
      2) It supports a short blend window + optional output rate limit
         to smooth the moment of switching.
    """

    SWING = "SWING"
    LQR = "LQR"

    def __init__(
        self,
        system,
        lqr_controller,
        swingup_controller,
        engage_angle_deg: float = 25.0,
        engage_speed_rad_s: float = 9.0,
        engage_cart_speed: float = 1.2,
        hold_time: float = 0.05,             # kept for compat, unused
        dropout_angle_deg: float = 45.0,
        dropout_speed_rad_s: float = 30.0,
        dropout_cart_speed: float = 10.0,
        allow_dropout: bool = True,
        min_time_in_mode: float = 0.10,      # kept for compat, unused
        verbose: bool = False,
        # smoothing
        blend_time: float = 0.12,
        du_max: Optional[float] = 800.0,
    ):
        self.sys = system
        self.c_lqr = lqr_controller
        self.c_swing = swingup_controller

        # Entry thresholds
        self.th_on = np.deg2rad(float(engage_angle_deg))
        self.w_on = float(engage_speed_rad_s)
        self.xd_on = float(engage_cart_speed)

        # Exit thresholds
        self.th_off = np.deg2rad(float(dropout_angle_deg))
        self.w_off = float(dropout_speed_rad_s)
        self.xd_off = float(dropout_cart_speed)
        self.allow_dropout = bool(allow_dropout)

        self.verbose = bool(verbose)

        # Mode state
        self.mode = self.SWING
        self._mode_since = 0.0
        self._last_log = -1.0

        # Switch smoothing
        self.blend_time = float(max(0.0, blend_time))
        self._blend_from_u = 0.0
        self._blend_t0 = 0.0
        self._blend_until = -1.0

        # Output rate limiting (helps avoid sharp reversals in animation)
        self.du_max = None if du_max is None else float(max(1e-3, du_max))
        self._u_last = 0.0
        self._t_last = None

    # -------------------------------------------------------------

    def _log(self, msg: str, t: float, every: float = 0.05) -> None:
        if self.verbose and ((self._last_log < 0.0) or (t - self._last_log >= every)):
            self._last_log = t
            print(msg)

    def _should_enter_lqr(self, theta_w: float, theta_dot: float, x_dot: float) -> bool:
        return (
            abs(theta_w) < self.th_on
            and abs(theta_dot) < self.w_on
            and abs(x_dot) < self.xd_on
        )

    def _should_exit_lqr(self, theta_w: float, theta_dot: float, x_dot: float) -> bool:
        if not self.allow_dropout:
            return False
        return (
            abs(theta_w) > self.th_off
            or abs(theta_dot) > self.w_off
            or abs(x_dot) > self.xd_off
        )

    @staticmethod
    def _clip(u: float, limit: Optional[float]) -> float:
        if limit is None:
            return float(u)
        lim = float(abs(limit))
        return float(np.clip(u, -lim, lim))

    def _rate_limit(self, t: float, u: float) -> float:
        if self.du_max is None:
            return float(u)
        if self._t_last is None:
            self._t_last = float(t)
            self._u_last = float(u)
            return float(u)

        t = float(t)
        if t <= self._t_last + 1e-12:
            return float(self._u_last)

        dt = t - self._t_last
        du_allowed = self.du_max * dt
        u_out = float(np.clip(u, self._u_last - du_allowed, self._u_last + du_allowed))
        self._t_last = t
        self._u_last = u_out
        return u_out

    def _start_blend(self, t: float, u_from: float) -> None:
        if self.blend_time <= 0.0:
            self._blend_until = -1.0
            return
        self._blend_from_u = float(u_from)
        self._blend_t0 = float(t)
        self._blend_until = float(t) + self.blend_time

    def _apply_blend(self, t: float, u_target: float) -> float:
        if self._blend_until < 0.0 or t >= self._blend_until:
            return float(u_target)
        alpha = (float(t) - self._blend_t0) / max(self.blend_time, 1e-9)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return float((1.0 - alpha) * self._blend_from_u + alpha * float(u_target))

    # -------------------------------------------------------------

    def cart_force(self, t: float, state) -> float:
        x, x_dot, theta, theta_dot = state
        theta_w = wrap_to_pi(float(theta))

        # Hard safety: never stay in LQR if too far tipped
        if self.mode == self.LQR and abs(theta_w) > (np.pi / 2):
            self.mode = self.SWING

        # Current output (for blending) = last output after rate limiting
        u_prev = float(self._u_last)

        # Mode transitions
        if self.mode == self.SWING:
            if self._should_enter_lqr(theta_w, float(theta_dot), float(x_dot)):
                if hasattr(self.c_lqr, "x_ref"):
                    self.c_lqr.x_ref = float(x)
                self.mode = self.LQR
                self._mode_since = float(t)
                self._start_blend(t, u_prev)
                self._log(f"[switch] ENTER LQR @ t={t:.3f}, x_ref={float(x):+.3f}", float(t))
        else:
            if self._should_exit_lqr(theta_w, float(theta_dot), float(x_dot)):
                self.mode = self.SWING
                self._mode_since = float(t)
                self._start_blend(t, u_prev)
                self._log(f"[switch] EXIT  LQR @ t={t:.3f}", float(t))

        # Controller selection + per-mode saturation
        if self.mode == self.LQR:
            u = float(self.c_lqr.cart_force(t, state))
            u = self._clip(u, getattr(self.c_lqr, "force_limit", None))
        else:
            u = float(self.c_swing.cart_force(t, state))
            swing_limit = getattr(self.c_swing, "F_max", getattr(self.c_swing, "force_limit", None))
            u = self._clip(u, swing_limit)

        # Smooth the switch moment
        u = self._apply_blend(float(t), u)

        # Optional final output rate limiting
        return self._rate_limit(float(t), u)
