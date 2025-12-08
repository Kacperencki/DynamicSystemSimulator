import numpy as np

def wrap_to_pi(th):
    return ((th + np.pi) % (2 * np.pi)) - np.pi


class SimpleSwitcher:
    """
    Simple 0/1 switcher between swing-up and LQR.

    Modes:
        - SWING : use swing-up controller
        - LQR   : use LQR controller

    Logic:
        - Start in SWING.
        - If in SWING and:
              |theta|   < engage_angle
              |theta_d| < engage_speed
              |x_dot|   < engage_cart_speed   (optional, can be large)
          → switch to LQR and latch current x as LQR.x_ref.
        - If in LQR and allow_dropout and:
              |theta|   > dropout_angle
              OR |theta_d| > dropout_speed
              OR |x_dot|   > dropout_cart_speed
          → switch back to SWING.

    The constructor keeps the old signature, but most timing / drift
    fancy stuff is removed for clarity.
    """

    SWING = "SWING"
    LQR = "LQR"

    def __init__(self,
                 system,
                 lqr_controller,
                 swingup_controller,
                 engage_angle_deg=25.0,
                 engage_speed_rad_s=9.0,
                 engage_cart_speed=1.2,
                 hold_time=0.05,             # kept for compat, currently unused
                 dropout_angle_deg=45.0,
                 dropout_speed_rad_s=30.0,
                 dropout_cart_speed=10.0,
                 allow_dropout=True,
                 min_time_in_mode=0.10,      # kept for compat, currently unused
                 verbose=False):

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

        # State
        self.mode = self.SWING
        self._mode_since = 0.0
        self._last_log = -1.0

    # -------------------------------------------------------------

    def _time_in_mode(self, t):  # kept for possible future use
        return t - self._mode_since

    def _log(self, msg, t, every=0.05):
        if self.verbose and ((self._last_log < 0.0) or (t - self._last_log >= every)):
            self._last_log = t
            print(msg)

    def _enter_lqr(self, t, x_now):
        # Latch x_ref if available
        if hasattr(self.c_lqr, "x_ref"):
            self.c_lqr.x_ref = float(x_now)
        self.mode = self.LQR
        self._mode_since = t
        self._log(f"[switch] ENTER LQR @ t={t:.3f}, x_ref={x_now:+.3f}", t)

    def _exit_lqr(self, t):
        self.mode = self.SWING
        self._mode_since = t
        self._log(f"[switch] EXIT  LQR @ t={t:.3f}", t)

    # -------------------------------------------------------------

    def _should_enter_lqr(self, theta_w, theta_dot, x_dot):
        """Simple "calm" condition."""
        return (
            abs(theta_w) < self.th_on
            and abs(theta_dot) < self.w_on
            and abs(x_dot) < self.xd_on
        )

    def _should_exit_lqr(self, theta_w, theta_dot, x_dot):
        if not self.allow_dropout:
            return False
        return (
            abs(theta_w) > self.th_off
            or abs(theta_dot) > self.w_off
            or abs(x_dot) > self.xd_off
        )

    # -------------------------------------------------------------

    def cart_force(self, t, state):
        x, x_dot, theta, theta_dot = state
        theta_w = wrap_to_pi(theta)

        # Hard safety: never stay in LQR if too far tipped
        if self.mode == self.LQR and abs(theta_w) > (np.pi / 2):
            self._exit_lqr(t)

        # Mode transitions
        if self.mode == self.SWING:
            if self._should_enter_lqr(theta_w, theta_dot, x_dot):
                self._enter_lqr(t, x)
        else:  # self.mode == self.LQR
            if self._should_exit_lqr(theta_w, theta_dot, x_dot):
                self._exit_lqr(t)

        # Control selection
        if self.mode == self.LQR:
            u = self.c_lqr.cart_force(t, state)
            self._log(
                f"[LQR ] t={t:6.3f} x_ref={getattr(self.c_lqr, 'x_ref', 0.0):+.3f} "
                f"θ={theta_w:+.3f} θ̇={theta_dot:+.2f} ẋ={x_dot:+.2f} u={u:+.2f}",
                t,
            )
        else:
            u = self.c_swing.cart_force(t, state)
            self._log(
                f"[SWUP] t={t:6.3f} θ={theta_w:+.3f} θ̇={theta_dot:+.2f} "
                f"ẋ={x_dot:+.2f} u={u:+.2f}",
                t,
            )

        # Final saturation based on LQR's limit (assumed to be present)
        limit = getattr(self.c_lqr, "force_limit", None)
        if limit is not None:
            return float(np.clip(u, -limit, limit))
        return float(u)
