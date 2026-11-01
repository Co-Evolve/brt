"""The brittle star: morphology, aquarium arena and the undirected locomotion, directed locomotion and light escape
tasks (registered as ``brittle_star/<task>``)."""

from biorobot.brittle_star.arena import AquariumArena, AquariumArenaConfiguration
from biorobot.brittle_star.environment import (
    DirectedLocomotionEnv,
    BrittleStarEnv,
    BrittleStarEnvConfig,
    LightEscapeEnv,
    UndirectedLocomotionEnv,
    DirectedLocomotionEnvConfig,
    LightEscapeEnvConfig,
    UndirectedLocomotionEnvConfig,
)
from biorobot.brittle_star.morphology import BrittleStarMorphology
from biorobot.brittle_star.specification import (
    BrittleStarActuationSpecification,
    BrittleStarArmSegmentSpecification,
    BrittleStarArmSpecification,
    BrittleStarDiskSpecification,
    BrittleStarJointSpecification,
    BrittleStarMorphologySpecification,
    BrittleStarSensorSpecification,
    default_brittle_star_morphology_specification,
)

__all__ = [
    "AquariumArena",
    "AquariumArenaConfiguration",
    "BrittleStarMorphology",
    "BrittleStarActuationSpecification",
    "BrittleStarArmSegmentSpecification",
    "BrittleStarArmSpecification",
    "BrittleStarDiskSpecification",
    "BrittleStarJointSpecification",
    "BrittleStarMorphologySpecification",
    "BrittleStarSensorSpecification",
    "default_brittle_star_morphology_specification",
    "BrittleStarEnv",
    "BrittleStarEnvConfig",
    "UndirectedLocomotionEnv",
    "UndirectedLocomotionEnvConfig",
    "DirectedLocomotionEnv",
    "DirectedLocomotionEnvConfig",
    "LightEscapeEnv",
    "LightEscapeEnvConfig",
]
