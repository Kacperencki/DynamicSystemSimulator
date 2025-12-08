import numpy as np
from typing import Optional


def wrap_to_pi(a: float) -> float:
    """Map any angle to (-π, π]."""
    return ((a + np.pi) % (2.0 * np.pi)) - np.pi


class AutoSwingUp:
    """
    Simplified energy-based swing-up controller for the cart–pole.

    Design:
        - State: [x, x_dot, theta, theta_dot]
          theta measured from the UPWARD vertical (theta=0 upright).

        - Energy bookkeeping uses bottom as zero:
              theta_up   = wrap_to_pi(theta)       (0 at upright)
              theta_down = wrap_to_pi(theta_up - π) (0 at bottom)

          Then:
              E = 0.5 * I_p * theta_dot^2 + m g l_c (1 - cos(theta_down))
              E_des ≈ 2 m g l_c   (upright ~ two "meters" above bottom)

        - Control law:

            Region 1 (far from upright, |theta_up| >= soft_zone):
                u_raw = ke * (E - E_des) * sign(theta_dot * cos(theta_up))
                        - kv * x_dot
                        - kx * x

            Region 2 (near upright, |theta_up| < soft_zone):
                u_raw = -soft_kx * x - soft_kv * x_dot

          In both regions u_raw is clipped to ±force_limit, then passed
          through a simple rate limiter to avoid stiff ODE steps.

    __init__ signature is compatible with older versions, but behaviour
    is mostly governed by:
        - ke          : energy gain
        - kv          : cart velocity damping
        - force_limit : max |u|
        - soft_zone_deg : half-width of capture zone around upright
    """

    def __init__(
        self,
        system,
        ke: Optional[float] = None,
        kv: Optional[float] = None,
        force_limit: Optional[float] = None,
        soft_zone_deg: float = 8.0,
        soft_kx: Optional[float] = None,
        soft_kv: Optional[float] = None,
        # shaping parameters (kept for API compatibility; not used in law)
        e_scale: float = 0.6,
        e_dead: float = 0.05,
        thd_clip: float = 12.0,
        # rate limiter
        du_max: float = 300.0,
    ):
        self.system = system

        m = float(system.m)
        M = float(system.M)
        lc = float(getattr(system, "lc", getattr(system, "l", 0.3)))

        # Aggressive but sane default energy gain for this plant scale
        # You can override via ke argument or GUI.
        self.ke = (3.0 / (m * lc)) if ke is None else float(ke)

        # Cart velocity damping
        self.kv = (1.5 * np.sqrt(M + m)) if kv is None else float(kv)

        # Force saturation
        self.F_max = 30.0 if force_limit is None else float(force_limit)

        # Capture zone around upright (in radians)
        self.soft_zone = np.deg2rad(float(soft_zone_deg))

        # PD capture gains in cart position / velocity near upright
        self.soft_kx = (2.0 * (M + m)) if soft_kx is None else float(soft_kx)
        self.soft_kv = (12.0 * np.sqrt(M + m)) if soft_kv is None else float(soft_kv)

        # Mild centering term used in pump region so the cart
        # doesn’t drift to infinity in one direction.
        self.kx = 0.5

        # Shaping parameters kept for compatibility (not used below)
        self.e_scale = float(max(1e-6, e_scale))
        self.e_dead = float(max(0.0, e_dead))
        self.thd_clip = float(max(1e-3, thd_clip))

        # Rate limiter state
        self.du_max = float(max(1e-3, du_max))
        self._last_u = 0.0
        self._last_t = None

    # ------------------------------------------------------------------
    # Energy bookkeeping (bottom = 0, upright ≈ 2 m g l_c)
    # ------------------------------------------------------------------

    def energy_desired(self) -> float:
        """Desired energy corresponding roughly to upright."""
        m = float(self.system.m)
        g = float(self.system.g)
        lc = float(getattr(self.system, "lc", getattr(self.system, "l", 0.3)))
        return 2.0 * m * g * lc

    def energy(self, state) -> float:
        """
        Mechanical energy measured from the *bottom* equilibrium.

        Uses:
            theta_up   = wrap_to_pi(theta)
            theta_down = wrap_to_pi(theta_up - pi)
        """
        _, _, th, thd = state
        th_up = wrap_to_pi(float(th))           # 0 at upright
        th_down = wrap_to_pi(th_up - np.pi)     # 0 at bottom

        Ip = float(self.system.Ip)              # inertia about pivot
        m = float(self.system.m)
        g = float(self.system.g)
        lc = float(getattr(self.system, "lc", getattr(self.system, "l", 0.3)))

        T = 0.5 * Ip * (float(thd) ** 2)
        V = m * g * lc * (1.0 - np.cos(th_down))
        return float(T + V)

    # ------------------------------------------------------------------
    # Main control law
    # ------------------------------------------------------------------

    def cart_force(self, t: float, state) -> float:
        x, x_dot, th, thd = state
        x = float(x)
        x_dot = float(x_dot)
        th_up = wrap_to_pi(float(th))
        thd = float(thd)

        # Region 1: capture near upright
        if abs(th_up) < self.soft_zone:
            u_raw = -self.soft_kx * x - self.soft_kv * x_dot
            u_raw = float(np.clip(u_raw, -self.F_max, self.F_max))
            return self._rate_limit(t, u_raw)

        # Region 2: energy pumping away from upright
        E = self.energy(state)
        Edes = self.energy_desired()
        Eerr = E - Edes

        # Classic direction term: sign(theta_dot * cos(theta_up))
        pump_dir = np.sign(thd * np.cos(th_up))
        if pump_dir == 0.0:
            pump_dir = 1.0

        # Energy pump + cart velocity damping + small centering on x
        u_raw = (
            self.ke * Eerr * pump_dir
            - self.kv * x_dot
            - self.kx * x
        )
        u_raw = float(np.clip(u_raw, -self.F_max, self.F_max))

        return self._rate_limit(t, u_raw)

    # ------------------------------------------------------------------
    # Helper: force rate limiter
    # ------------------------------------------------------------------

    def _rate_limit(self, t: float, u_cmd: float) -> float:
        """
        Limit |du/dt| to avoid stiff ODE steps.
        """
        if self._last_t is None:
            self._last_t = float(t)
            self._last_u = float(u_cmd)
            return self._last_u

        dt = float(max(1e-4, t - self._last_t))
        du_allowed = self.du_max * dt
        u = float(
            np.clip(
                u_cmd,
                self._last_u - du_allowed,
                self._last_u + du_allowed,
            )
        )

        if not np.isfinite(u):
            u = 0.0

        self._last_t = float(t)
        self._last_u = u
        return u
