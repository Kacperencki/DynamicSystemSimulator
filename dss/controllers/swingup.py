import numpy as np
from typing import Optional

from dss.utils.angles import wrap_to_pi


class AutoSwingUp:
    """
    Simplified energy-based swing-up controller for the cart–pole.

    State: [x, x_dot, theta, theta_dot]
    theta measured from the UPWARD vertical (theta=0 upright).

    NOTE:
    Adaptive ODE solvers may call the controller at the same t or with decreasing t
    (rejected steps). The internal rate limiter must be safe under that.
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
        # shaping parameters
        e_scale: float = 0.6,
        e_dead: float = 0.05,
        thd_clip: float = 12.0,
        dir_k: float = 6.0,
        # rate limiter
        du_max: float = 300.0,
    ):
        self.system = system

        m = float(system.m)
        M = float(system.M)
        lc = float(getattr(system, "lc", getattr(system, "l", 0.3)))

        self.ke = (3.0 / (m * lc)) if ke is None else float(ke)
        self.kv = (1.5 * np.sqrt(M + m)) if kv is None else float(kv)

        self.F_max = 30.0 if force_limit is None else float(force_limit)

        self.soft_zone = np.deg2rad(float(soft_zone_deg))
        self.soft_kx = (2.0 * (M + m)) if soft_kx is None else float(soft_kx)
        self.soft_kv = (12.0 * np.sqrt(M + m)) if soft_kv is None else float(soft_kv)

        self.kx = 0.5

        self.e_scale = float(max(1e-6, e_scale))
        self.e_dead = float(max(0.0, e_dead))
        self.thd_clip = float(max(1e-3, thd_clip))
        self.dir_k = float(max(1e-3, dir_k))

        self.du_max = float(max(1e-3, du_max))
        self._last_u = 0.0
        self._last_t = None

    # ------------------------------------------------------------------
    # Energy bookkeeping (bottom = 0, upright ≈ 2 m g l_c)
    # ------------------------------------------------------------------

    def energy_desired(self) -> float:
        m = float(self.system.m)
        g = float(self.system.g)
        lc = float(getattr(self.system, "lc", getattr(self.system, "l", 0.3)))
        return 2.0 * m * g * lc

    def energy(self, state) -> float:
        _, _, th, thd = state
        th_up = wrap_to_pi(float(th))           # 0 at upright
        th_down = wrap_to_pi(th_up - np.pi)     # 0 at bottom

        Ip = float(self.system.Ip)
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

        En = float(E - Edes) / float(max(1e-6, Edes))
        if abs(En) < self.e_dead:
            En = 0.0
        Eterm = float(np.tanh(En / self.e_scale))

        thd_use = float(np.clip(thd, -self.thd_clip, self.thd_clip))
        c = float(np.cos(th_up))
        phase = thd_use * c

        if abs(phase) < 1e-6:
            pump_dir = 1.0
        else:
            pump_dir = float(np.tanh(self.dir_k * phase))

        u_raw = (
            self.ke * Eterm * pump_dir
            - self.kv * x_dot
            - self.kx * x
        )
        u_raw = float(np.clip(u_raw, -self.F_max, self.F_max))

        return self._rate_limit(t, u_raw)

    # ------------------------------------------------------------------
    # Helper: force rate limiter (adaptive-solver safe)
    # ------------------------------------------------------------------

    def _rate_limit(self, t: float, u_cmd: float) -> float:
        t = float(t)
        u_cmd = float(u_cmd)
        eps = 1e-12

        if self._last_t is None:
            self._last_t = t
            self._last_u = u_cmd
            return self._last_u

        # Same-time or backwards-time evaluations can happen in adaptive solvers.
        # Do not "invent" a dt; just return the current command and do not update state.
        if t <= self._last_t + eps:
            return u_cmd

        dt = t - self._last_t
        du_allowed = self.du_max * dt
        u = float(np.clip(u_cmd, self._last_u - du_allowed, self._last_u + du_allowed))

        if not np.isfinite(u):
            u = 0.0

        self._last_t = t
        self._last_u = u
        return u
