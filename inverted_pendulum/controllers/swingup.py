import numpy as np

def wrap_to_pi(a: float) -> float:
    """Map any angle to (-π, π]."""
    return ((a + np.pi) % (2 * np.pi)) - np.pi


class AutoSwingUp:
    """
    Energy-based swing-up with plant-scaled defaults + soft shapers
    and a small **rate limiter** to avoid solver stiffness freezes.
    """

    def __init__(self, system,
                 ke: float | None = None,
                 kv: float | None = None,
                 force_limit: float | None = None,
                 soft_zone_deg: float = 70.0,
                 soft_kx: float | None = None,
                 soft_kv: float | None = None,
                 # shaping parameters
                 e_scale: float = 0.6,        # fraction of E* used to scale tanh()
                 e_dead: float = 0.05,        # dead-zone as fraction of E*
                 thd_clip: float = 12.0,      # soft limit for |theta_dot| in pump [rad/s]
                 # NEW: rate limiter (prevents stiffness/freezes)
                 du_max: float = 300.0        # max |dF/dt| in N/s (tune per plant)
                 ):

        self.system = system
        m, M = float(system.m), float(system.M)
        lc = float(getattr(system, "lc", getattr(system, "l", 0.3)))

        # Slightly gentler defaults (reduce violent kicks)
        self.ke    = (3.0 / (m * lc))           if ke is None else float(ke)   # was 4.0/(m*lc)
        self.kv    = (1.5 * np.sqrt(M + m))     if kv is None else float(kv)   # was 2.0*sqrt()
        self.F_max = (30.0)                     if force_limit is None else float(force_limit)

        self.soft_zone = np.deg2rad(float(soft_zone_deg))  # radians

        # Gentle capture gains
        self.soft_kx = (2.0 * (M + m))          if soft_kx is None else float(soft_kx)
        self.soft_kv = (12.0 * np.sqrt(M + m))  if soft_kv is None else float(soft_kv)

        # shaper params
        self.e_scale  = float(max(1e-6, e_scale))
        self.e_dead   = float(max(0.0, e_dead))
        self.thd_clip = float(max(1e-3, thd_clip))

        # rate limiter
        self.du_max   = float(max(1e-3, du_max))
        self._last_u  = 0.0
        self._last_t  = None

    # -------- energy bookkeeping (bottom = 0, upright = 2 m g lc) --------
    def energy_desired(self) -> float:
        m, g = float(self.system.m), float(self.system.g)
        lc   = float(getattr(self.system, "lc", getattr(self.system, "l", 0.3)))
        return 2.0 * m * g * lc

    def energy(self, state) -> float:
        _, _, th, thd = state
        th_up   = wrap_to_pi(float(th))            # 0 at upright
        th_down = wrap_to_pi(th_up - np.pi)        # 0 at bottom

        Ip = float(self.system.Ip)                 # inertia about pivot
        m  = float(self.system.m)
        g  = float(self.system.g)
        lc = float(getattr(self.system, "lc", getattr(self.system, "l", 0.3)))

        T = 0.5 * Ip * (float(thd) ** 2)
        V = m * g * lc * (1.0 - np.cos(th_down))
        return float(T + V)

    # -------- control law --------
    def cart_force(self, t: float, state) -> float:
        x, x_dot, th, thd = state
        th_up = wrap_to_pi(float(th))

        # --- soft capture near upright ---
        if abs(th_up) < self.soft_zone:
            u_raw = - self.soft_kx * float(x) - self.soft_kv * float(x_dot)
            u_raw = float(np.clip(u_raw, -self.F_max, self.F_max))
            return self._rate_limit(t, u_raw)

        # --- energy pump away from upright ---
        E    = self.energy(state)
        Edes = max(1e-9, self.energy_desired())   # guard
        Eerr = E - Edes

        # energy dead-zone (prevents over-pumping when close)
        if abs(Eerr) <= self.e_dead * Edes:
            u_raw = - self.soft_kx * float(x) - self.soft_kv * float(x_dot)
            u_raw = float(np.clip(u_raw, -self.F_max, self.F_max))
            return self._rate_limit(t, u_raw)

        # smooth-bounded energy error and angular speed
        E_scale = self.e_scale * Edes
        Eerr_bounded = E_scale * np.tanh(Eerr / E_scale)

        thd_eff = self.thd_clip * np.tanh(float(thd) / self.thd_clip)

        pump = thd_eff * np.cos(th_up)
        u_raw = self.ke * Eerr_bounded * pump - self.kv * float(x_dot)
        u_raw = float(np.clip(u_raw, -self.F_max, self.F_max))

        return self._rate_limit(t, u_raw)

    # -------- helper: force rate limiter --------
    def _rate_limit(self, t: float, u_cmd: float) -> float:
        """Limit |du/dt| to avoid stiff ODE steps (freezes)."""
        if self._last_t is None:
            self._last_t = float(t)
            self._last_u = float(u_cmd)
            return self._last_u

        dt = float(max(1e-4, t - self._last_t))  # protect very small/zero dt
        du_allowed = self.du_max * dt
        u = float(np.clip(u_cmd, self._last_u - du_allowed, self._last_u + du_allowed))

        # NaN/Inf guard
        if not np.isfinite(u):
            u = 0.0

        self._last_t = float(t)
        self._last_u = u
        return u
