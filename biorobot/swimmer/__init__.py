"""The swimmer: an abstract anguilliform body in a viscous fluid, and the head-first swimming task (registered as
``swimmer/head_first_swimming``).

The swimmer uses MuJoCo's ellipsoid fluid model by default, which ``backend="mjx"`` does not support (MJX's JAX
implementation silently replaces it by the inertia-based model); use ``"mjc"`` or ``"warp"``.
"""

from biorobot.swimmer.arena import FluidArena, FluidArenaConfiguration
from biorobot.swimmer.environment import (
    HeadFirstSwimmingEnv,
    HeadFirstSwimmingEnvConfig,
)
from biorobot.swimmer.morphology import SwimmerMorphology
from biorobot.swimmer.specification import (
    REVIEW_JOINT_RANGE,
    SwimmerActuationSpecification,
    SwimmerJointSpecification,
    SwimmerMorphologySpecification,
    SwimmerSegmentSpecification,
    default_swimmer_morphology_specification,
    segment_lengths,
)

__all__ = [
    "REVIEW_JOINT_RANGE",
    "FluidArena",
    "FluidArenaConfiguration",
    "HeadFirstSwimmingEnv",
    "HeadFirstSwimmingEnvConfig",
    "SwimmerMorphology",
    "SwimmerActuationSpecification",
    "SwimmerJointSpecification",
    "SwimmerMorphologySpecification",
    "SwimmerSegmentSpecification",
    "default_swimmer_morphology_specification",
    "segment_lengths",
]
