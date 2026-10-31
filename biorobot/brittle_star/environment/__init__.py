"""Brittle star tasks. Importing this package registers them with :func:`biorobot.make`."""

from biorobot.brittle_star.environment.base import BrittleStarEnv, BrittleStarEnvConfig
from biorobot.brittle_star.environment.directed_locomotion import (
    DirectedLocomotionEnv,
    DirectedLocomotionEnvConfig,
)
from biorobot.brittle_star.environment.light_escape import (
    LightEscapeEnv,
    LightEscapeEnvConfig,
)
from biorobot.brittle_star.environment.undirected_locomotion import (
    UndirectedLocomotionEnv,
    UndirectedLocomotionEnvConfig,
)

__all__ = [
    "BrittleStarEnv",
    "BrittleStarEnvConfig",
    "DirectedLocomotionEnv",
    "DirectedLocomotionEnvConfig",
    "LightEscapeEnv",
    "LightEscapeEnvConfig",
    "UndirectedLocomotionEnv",
    "UndirectedLocomotionEnvConfig",
]
