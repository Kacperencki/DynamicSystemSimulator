from __future__ import annotations
import numpy as np


class MotorWrapper:
    """
    Couples a DC motor to a mechanical model through a gear train.

    The motor drives the plant via a gearbox with ratio N (motor turns / plant turn)
    and efficiency eta. At each solver step the wrapper:

      1. Reads the plant's joint speed and scales it to motor shaft speed (omega_m = N * omega_port).
      2. Integrates the motor's electrical ODE (di/dt) using the back-EMF at that speed.
      3. Converts motor current to torque, applies gear ratio and efficiency, and passes
         the resulting torque to the plant as its control input.

    The combined state vector is [i, *plant_state], where i is motor current [A].
    """

    def __init__(self, model, motor, N: float = 8.0, eta: float = 0.9, reflect: bool = False):
        """
        Parameters
        ----------
        model : DynamicalSystem
            The mechanical plant being driven (e.g. Pendulum). Must expose
            joint_speed(state) and dynamics(t, state, torque).
        motor : DCMotor
            Motor model supplying electrical dynamics (R, L, Ke, Kt, Im, bm).
        N : float
            Gear ratio — motor shaft turns per one plant joint turn.
            Higher N gives more torque but less speed at the plant.
        eta : float
            Gear efficiency in (0, 1]. Accounts for friction losses in the gearbox;
            delivered torque = eta * N * Kt * i.
        reflect : bool
            If True, the motor's rotor inertia (Im) and viscous damping (bm) are
            reflected through the gear ratio and added to the plant's inertia and
            damping terms before simulation. This is the standard lumped-parameter
            approach when the motor inertia is non-negligible relative to the load.
        """
        self.model = model
        self.motor = motor
        self.N = float(N)
        self.eta = float(eta)
        self.reflect = bool(reflect)

        if self.reflect:
            # Reflect rotor inertia and damping to the plant side:
            # I_reflected = Im / N^2,  b_reflected = bm / N^2
            self.model.I += self.motor.Im / (self.N**2)
            self.model.b += self.motor.bm / (self.N**2)



    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        i = state[0]
        other_state = state[1:]


        # 1) plant joint speed (for back-EMF)
        omega_port = self.model.joint_speed(other_state)
        omega_m = self.N * omega_port

        # 2) motor electrical ODE
        V = self.motor.voltage_func(t) if callable(self.motor.voltage_func) else float(self.motor.voltage_func)
        di_dt = (V - self.motor.R * i - self.motor.Ke * omega_m) / self.motor.L

        # 3) torque delivered to plant through gear
        tau_drive = self.eta * self.N * (self.motor.Kt * i)

        other_state_dot = self.model.dynamics(t, other_state, tau_drive)


        return np.concatenate(([di_dt], other_state_dot))

    def state_labels(self) -> list[str]:
        return ["i [A]"] + (self.model.state_labels())

    def positions(self, state: np.ndarray) -> list:
        state = state[1:]
        return self.model.positions(state)