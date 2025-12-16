import numpy as np


class DCMotor:
    """
    DC motor model used by the Streamlit runner.

    Electrical:
        L di/dt = v(t) - R*i - Ke*omega
    Mechanical:
        J domega/dt = Kt*i - bm*omega - tau_load(t, omega)

    Supports:
      - runner-style configuration: v_mode/V0/v_offset/t_step/v_freq/v_duty and load_mode/tau_load/b_load/tau_c/omega_eps
      - legacy usage: voltage_func (callable or scalar) and load_func (callable or scalar)
      - inertia passed as J=... or Im=... (alias)
    """

    def __init__(
        self,
        R,
        L,
        Ke,
        Kt,
        J=None,
        bm=0.0,
        Im=None,
        *,
        voltage_func=None,
        load_func=None,
        v_mode="constant",
        V0=6.0,
        v_offset=0.0,
        t_step=0.0,
        v_freq=1.0,
        v_duty=0.5,
        load_mode="none",
        tau_load=0.0,
        b_load=0.0,
        tau_c=0.0,
        omega_eps=0.5,
        **_ignored,
    ):
        self.R = float(R)      # [ohm]
        self.L = float(L)      # [H]
        self.Ke = float(Ke)    # [V/(rad/s)]
        self.Kt = float(Kt)    # [N*m/A]

        if J is None and Im is None:
            raise TypeError("Provide motor inertia as J=... (preferred) or Im=...")

        self.J = float(J if J is not None else Im)  # [kg*m^2]
        self.Im = self.J  # legacy alias

        self.bm = float(bm)    # [N*m*s/rad]

        # Voltage profile params (runner-style)
        self.v_mode = str(v_mode)
        self.V0 = float(V0)
        self.v_offset = float(v_offset)
        self.t_step = float(t_step)
        self.v_freq = float(v_freq)
        self.v_duty = float(v_duty)

        # Load model params (runner-style)
        self.load_mode = str(load_mode)
        self.tau_load = float(tau_load)
        self.b_load = float(b_load)
        self.tau_c = float(tau_c)
        self.omega_eps = float(omega_eps)

        # Legacy hooks
        self._user_voltage_func = voltage_func
        self._user_load_func = load_func

        # Compatibility attributes used elsewhere (e.g. wrappers)
        self.voltage_func = voltage_func if voltage_func is not None else self.voltage
        self.load_func = load_func if load_func is not None else self.load

    # --- Profiles -------------------------------------------------

    def voltage(self, t: float) -> float:
        """Return v(t) [V]. Uses voltage_func if provided, else uses v_mode params."""
        if self._user_voltage_func is not None:
            return self._user_voltage_func(t) if callable(self._user_voltage_func) else float(self._user_voltage_func)

        off = self.v_offset
        V0 = self.V0
        mode = self.v_mode

        if mode == "constant":
            return off + V0
        if mode == "step":
            return off + V0 * (1.0 if t >= self.t_step else 0.0)
        if mode == "ramp":
            return off + V0 * max(t - self.t_step, 0.0)
        if mode == "sine":
            return off + V0 * float(np.sin(2.0 * np.pi * self.v_freq * t))
        if mode == "square":
            phase = (self.v_freq * t) % 1.0
            return off + V0 * (1.0 if phase < self.v_duty else 0.0)

        # fallback
        return off + V0

    def load(self, t: float, omega: float) -> float:
        """Return tau_load(t, omega) [N*m]. Uses load_func if provided, else uses load_mode params."""
        if self._user_load_func is not None:
            return self._user_load_func(t, omega) if callable(self._user_load_func) else float(self._user_load_func)

        mode = self.load_mode
        if mode == "none":
            return 0.0
        if mode == "constant":
            return self.tau_load
        if mode == "viscous":
            return self.b_load * omega
        if mode == "coulomb":
            # smooth Coulomb friction (avoids discontinuity at omega=0)
            eps = max(self.omega_eps, 1e-12)
            return self.tau_c * float(np.tanh(omega / eps))

        return 0.0

    # --- Core model ------------------------------------------------

    def dynamics(self, t, state):
        i, omega = float(state[0]), float(state[1])

        V = self.voltage(t)
        tau_load = self.load(t, omega)

        di_dt = (V - self.R * i - self.Ke * omega) / self.L
        domega_dt = (self.Kt * i - self.bm * omega - tau_load) / self.J

        return np.array([di_dt, domega_dt], dtype=float)

    def energy_check(self, state):
        i, omega = float(state[0]), float(state[1])
        E_ind = 0.5 * self.L * i**2
        E_kin = 0.5 * self.J * omega**2
        E_tot = E_ind + E_kin
        return np.array([E_ind, E_kin, E_tot], dtype=float)

    def state_labels(self):
        return ["i [A]", "omega [rad/s]"]
