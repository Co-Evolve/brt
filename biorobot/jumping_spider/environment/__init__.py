"""The jumping spider tasks (importing this module registers them with :func:`biorobot.make`)."""

from biorobot.jumping_spider.environment.base import (
    JumpingSpiderEnv,
    JumpingSpiderEnvConfig,
)
from biorobot.jumping_spider.environment.directed_jump import (
    DirectedJumpEnv,
    DirectedJumpEnvConfig,
    make_directed_jump_env,
)
from biorobot.jumping_spider.environment.long_jump import (
    LongJumpEnv,
    LongJumpEnvConfig,
    make_long_jump_env,
)
from biorobot.jumping_spider.environment.platform_jump import (
    PlatformJumpEnv,
    PlatformJumpEnvConfig,
    make_platform_jump_env,
)

__all__ = [
    "JumpingSpiderEnv",
    "JumpingSpiderEnvConfig",
    "DirectedJumpEnv",
    "DirectedJumpEnvConfig",
    "make_directed_jump_env",
    "LongJumpEnv",
    "LongJumpEnvConfig",
    "make_long_jump_env",
    "PlatformJumpEnv",
    "PlatformJumpEnvConfig",
    "make_platform_jump_env",
]
