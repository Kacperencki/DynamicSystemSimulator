# swingup.py  — replace the class with this version

import numpy as np

def wrap_to_pi(theta):
    return ((theta + np.pi) % (2*np.pi)) - np.pi  # (-π, π]

class EnergySwingUp:
    def __init__(self, system,
                 ke=4.0,            # was 8.0 → calmer pumping
                 kv=3.5,            # was 2.0 → more cart damping out of soft zone
                 force_limit=20.0,  # was 30 → temporarily lower to reduce crazy speeds
                 soft_zone_deg=45.0,# was 30 → start calming earlier
                 soft_kx=1.0,       # NEW: gentle recentering in soft zone
                 soft_kv=6.0):      # NEW: strong braking in soft zone
        self.system = system
        self.ke = ke
        self.kv = kv
        self.F_max = force_limit
        self.soft_zone = np.deg2rad(soft_zone_deg)
        self.soft_kx = soft_kx
        self.soft_kv = soft_kv

    def energy_desired(self):  # E_upright relative to bottom (θ=π is bottom)
        return 2.0 * self.system.m * self.system.g * self.system.l

    def energy(self, state):
        _, _, theta, theta_dot = state
        theta_up = wrap_to_pi(theta)               # 0 at upright
        theta_down = wrap_to_pi(theta_up - np.pi)  # 0 at bottom
        m, l, g = self.system.m, self.system.l, self.system.g
        T = 0.5 * m * l**2 * theta_dot**2
        V = m * g * l * (1 - np.cos(theta_down))   # 0 at bottom, 2mgl at upright
        return T + V

    def cart_force(self, t, state):
        x, x_dot, theta, theta_dot = state
        theta_up = wrap_to_pi(theta)

        # --- Soft capture zone near upright: stop pumping, just brake & recenter ---
        if abs(theta_up) < self.soft_zone:
            u = - self.soft_kx * x - self.soft_kv * x_dot
            return float(np.clip(u, -self.F_max, self.F_max))

        # --- Energy shaping elsewhere (pumping) ---
        E = self.energy(state)
        Edes = self.energy_desired()
        pump = theta_dot * np.cos(theta_up)
        u = self.ke * (E - Edes) * pump - self.kv * x_dot
        return float(np.clip(u, -self.F_max, self.F_max))

