"""Swimmer tasks. Importing this package registers them with :func:`biorobot.make`."""

from biorobot.swimmer.environment.head_first_swimming import (
    HeadFirstSwimmingEnv,
    HeadFirstSwimmingEnvConfig,
    make_head_first_swimming_env,
)

__all__ = [
    "HeadFirstSwimmingEnv",
    "HeadFirstSwimmingEnvConfig",
    "make_head_first_swimming_env",
]
