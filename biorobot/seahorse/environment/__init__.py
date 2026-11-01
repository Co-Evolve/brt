"""The seahorse tail environment (importing this module registers it with :func:`biorobot.make`)."""

from biorobot.seahorse.environment.tail import (
    SeahorseTailEnv,
    SeahorseTailEnvConfig,
    default_tail_morphology_specification,
    make_tail_env,
)

__all__ = [
    "SeahorseTailEnv",
    "SeahorseTailEnvConfig",
    "default_tail_morphology_specification",
    "make_tail_env",
]
