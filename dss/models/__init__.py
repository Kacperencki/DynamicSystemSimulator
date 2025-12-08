# models/__init__.py

from .pendulum import Pendulum
from .double_pendulum import DoublePendulum
from .dc_motor import DCMotor
from dss.wrappers.motor_wrapper import MotorWrapper
from .vanderpoll_circuit import VanDerPol

# Inverted pendulum lives in a separate package, but we expose it here
from dss.models.inverted_pendulum import InvertedPendulum


def make_pendulum(mode: str, **kwargs):
    return Pendulum(mode=mode, **kwargs)


def make_double_pendulum(mode: str, **kwargs):
    return DoublePendulum(mode=mode, **kwargs)


def make_inverted_pendulum(mode: str, **kwargs):
    return InvertedPendulum(mode=mode, **kwargs)


MODEL_REGISTRY = {
    "pendulum": make_pendulum,
    "double_pendulum": make_double_pendulum,
    "inverted_pendulum": make_inverted_pendulum,
    "dc_motor": lambda mode="default", **kw: DCMotor(**kw),
    "motor_wrapper": lambda mode="default", **kw: MotorWrapper(**kw),
    "vanderpol": lambda mode="default", **kw: VanDerPol(**kw),
}


def get_model(model_name: str, mode: str, **kwargs):
    try:
        factory = MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Available: {list(MODEL_REGISTRY)}")
    return factory(mode=mode, **kwargs)
