"""The jumping spider (Salticidae): long jump, platform jump and directed jump tasks.

Create an environment with ``biorobot.make("jumping_spider/long_jump", backend="mjc")`` (or ``"mjx"``/``"warp"``).
"""

from biorobot.jumping_spider.arenas import (
    DirectedJumpArena,
    DirectedJumpArenaConfiguration,
    JumpingSpiderArena,
    JumpingSpiderArenaConfiguration,
    LongJumpArena,
    LongJumpArenaConfiguration,
    PlatformJumpArena,
    PlatformJumpArenaConfiguration,
)
from biorobot.jumping_spider.environment import (
    DirectedJumpEnv,
    DirectedJumpEnvConfig,
    JumpingSpiderEnv,
    JumpingSpiderEnvConfig,
    LongJumpEnv,
    LongJumpEnvConfig,
    PlatformJumpEnv,
    PlatformJumpEnvConfig,
)
from biorobot.jumping_spider.morphology import JumpingSpiderMorphology
from biorobot.jumping_spider.specification import (
    AbdomenSpecification,
    ActuationSpecification,
    CephalothoraxSpecification,
    DraglineSpecification,
    JointSpecification,
    JumpingSpiderMorphologySpecification,
    LegSegmentSpecification,
    LegSpecification,
    default_jumping_spider_morphology_specification,
)

__all__ = [
    "DirectedJumpArena",
    "DirectedJumpArenaConfiguration",
    "JumpingSpiderArena",
    "JumpingSpiderArenaConfiguration",
    "LongJumpArena",
    "LongJumpArenaConfiguration",
    "PlatformJumpArena",
    "PlatformJumpArenaConfiguration",
    "DirectedJumpEnv",
    "DirectedJumpEnvConfig",
    "JumpingSpiderEnv",
    "JumpingSpiderEnvConfig",
    "LongJumpEnv",
    "LongJumpEnvConfig",
    "PlatformJumpEnv",
    "PlatformJumpEnvConfig",
    "JumpingSpiderMorphology",
    "AbdomenSpecification",
    "ActuationSpecification",
    "CephalothoraxSpecification",
    "DraglineSpecification",
    "JointSpecification",
    "JumpingSpiderMorphologySpecification",
    "LegSegmentSpecification",
    "LegSpecification",
    "default_jumping_spider_morphology_specification",
]
