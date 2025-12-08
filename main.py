# main.py (simple toggle version)

from dss.core.simulator import Simulator
from dss.models import Pendulum, DoublePendulum, VanDerPol, DCMotor, MotorWrapper
import numpy as np

# If you actually use the inverted-pendulum stack, uncomment these:
from dss.controllers.lqr_controller import AutoLQR
from dss.controllers.swingup import AutoSwingUp
from dss.controllers import AutoSwitcher
from dss.wrappers.closed_lood_cart import CloseLoopCart
from dss.models.inverted_pendulum import InvertedPendulum

# =======================
# PICK WHAT TO SIMULATE:
# =======================
SYSTEM = "inverted"   # options: "pendulum", "double", "vdp", "inverted"
MODE = "ideal"    # for pendulum/double: "ideal", "damped", "driven", "dc_driven"


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
80
def build_inverted():
    """
    Inverted pendulum (cart–pole) with auto swing-up → LQR handoff.
    - Plant runs with damping so physics are realistic.
    - Controllers auto-tune from the current plant parameters, but remain tweakable.
    - Initial state is pendulum DOWN (theta=pi) so swing-up has something to do.
    """
    # Pick a sensible plant mode (accept external actuation in all modes).
    inv_mode = {
        "ideal":        "ideal",
        "damped_cart":  "damped_cart",
        "damped_pend":  "damped_pend",
        "damped_both":  "damped_both",
        "driven":       "driven",
        "dc_driven":    "damped_both",  # dc_driven ≈ damped plant + external inputs
    }.get(MODE, "damped_both")

    inv = InvertedPendulum(
        mode=inv_mode,
        l=0.30, m=0.20, cart_mass=0.50, g=9.81,
        mass_model="point",      # try "point" vs "uniform"
        b_cart=0.12, b_pend=0.02,  # viscous friction (Coulomb optional via coulomb_cart/coulomb_pend)
    )

    # Auto controllers (good defaults; still tweakable)
    lqr   = AutoLQR(inv)                    # or AutoLQR(inv, Q=..., R=..., u_max=...)
    swing = AutoSwingUp(inv)                # or AutoSwingUp(inv, ke=..., kv=..., force_limit=...)
    policy = AutoSwitcher(inv, lqr, swing)

    # Close the loop (controller → plant)
    cart = CloseLoopCart(system=inv, controller=policy)

    # Start with pendulum DOWN
    x0 = np.array([0.0, 0.0, np.pi, 0.0], dtype=float)  # [x, x_dot, theta, theta_dot]
    return cart, x0

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
