from __future__ import annotations
import numpy as np


class MotorWrapper:

    def __init__(self, model, motor, N=8.0, eta=0.9, reflect=False):
        self.model = model
        self.motor = motor
        self.N = float(N)
        self.eta = float(eta)
        self.reflect = bool(reflect)


        if self.reflect:
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