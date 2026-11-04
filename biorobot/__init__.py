"""The Bio-inspired Robotics Testbed (BRT): MuJoCo environments of animals for robotics and biology."""

__version__ = "1.1.0"

from biorobot.registry import available, make, register

__all__ = ["available", "make", "register", "__version__"]
