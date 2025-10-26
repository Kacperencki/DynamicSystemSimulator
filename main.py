# main.py (simple toggle version)

from core.simulator import Simulator
from systems import Pendulum, DoublePendulum, VanDerPol, DCMotor, MotorWrapper
import numpy as np

# If you actually use the inverted-pendulum stack, uncomment these:
from inverted_pendulum.controllers.lqr_controller import LQRController
from inverted_pendulum.controllers.swingup import EnergySwingUp
from inverted_pendulum.controllers.simple_switcher import SimpleSwitcher
from inverted_pendulum.closed_lood_cart import CloseLoopCart
from inverted_pendulum.inverted_pendulum import InvertedPendulum

# =======================
# PICK WHAT TO SIMULATE:
# =======================
SYSTEM = "inverted"   # options: "pendulum", "double", "vdp", "inverted"
MODE = "dc_driven"        # for pendulum/double: "ideal", "damped", "driven", "dc_driven"


# Quick params you can tweak
PEND = dict(length=0.12, mass=0.25, damping=1e-3)  # single pendulum
DP   = dict(length1=0.12, mass1=0.25, length2=0.12, mass2=0.25, damping1=1e-3, damping2=1e-3)  # double

DRIVE = dict(amplitude=0.05, frequency=4.0, phase=0.0)  # used for "driven" mode

MOTOR = dict(  # DC motor params (used for dc_driven modes)
    R=3.2, L=2.5e-3, Ke=0.05, Kt=0.05, Im=1.2e-5, bm=1e-4, voltage_func=6.0, load_func=0.0
)

def make_motor():
    return DCMotor(**MOTOR)

def build_pendulum():
    pend = Pendulum(
        length=PEND["length"], mass=PEND["mass"], mode=MODE, damping=PEND["damping"],
        drive_amplitude=DRIVE["amplitude"], drive_frequency=DRIVE["frequency"], drive_phase=DRIVE["phase"]
    )
    if MODE == "dc_driven":
        motor = make_motor()
        system = MotorWrapper(model=pend, motor=motor, reflect=True)  # 1-DOF -> True
        x0 = [0.0, np.pi/2, 0.0]  # [i, theta, theta_dot]
    else:
        system = pend
        x0 = [np.pi/2, 0.0]       # [theta, theta_dot]
    return system, np.array(x0, float)

def build_double():
    dp = DoublePendulum(
        length1=DP["length1"], mass1=DP["mass1"], length2=DP["length2"], mass2=DP["mass2"],
        mode=MODE, damping1=DP["damping1"], damping2=DP["damping2"],
        drive_amplitude=DRIVE["amplitude"], drive_frequency=DRIVE["frequency"], drive_phase=DRIVE["phase"],
        gravity=9.81,
    )
    if MODE == "dc_driven":
        motor = make_motor()
        system = MotorWrapper(model=dp, motor=motor, reflect=False)  # 2-DOF -> False
        x0 = [0.0, np.pi/2, 0.0, np.pi/4, 0.0]  # [i, th1, th1dot, th2, th2dot]
    else:
        system = dp
        x0 = [np.pi/2, 0.0, np.pi/4, 0.0]       # [th1, th1dot, th2, th2dot]
    return system, np.array(x0, float)

def build_vdp():
    van = VanDerPol()
    x0 = [1.0, 0.0]
    return van, np.array(x0, float)

def build_inverted():
    # Uncomment the imports at the top if you use this
    inv = InvertedPendulum(mode="ideal")
    lqr = LQRController(system=inv)
    swing = EnergySwingUp(system=inv)
    switch = SimpleSwitcher(system=inv, lqr_controller=lqr, swingup_controller=swing)
    cart = CloseLoopCart(system=inv, controller=switch)
    x0 = [0.0, 0.0, np.pi, 0.0]
    return cart, np.array(x0, float)

def pick():
    if SYSTEM == "pendulum": return build_pendulum()
    if SYSTEM == "double":   return build_double()
    if SYSTEM == "vdp":      return build_vdp()
    if SYSTEM == "inverted": return build_inverted()
    raise ValueError(f"Unknown SYSTEM='{SYSTEM}'")

def main():
    system, x0 = pick()
    simul = Simulator(system=system, initial_conditions=x0)
    simul.simulate()

if __name__ == "__main__":
    main()
