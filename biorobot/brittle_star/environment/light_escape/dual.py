from moojoco.environment.dual import DualMuJoCoEnvironment

from biorobot.brittle_star.environment.light_escape.mjc_env import (
    BrittleStarLightEscapeMJCEnvironment,
)
from biorobot.brittle_star.environment.light_escape.mjx_env import (
    BrittleStarLightEscapeMJXEnvironment,
)


class BrittleStarLightEscapeEnvironment(DualMuJoCoEnvironment):
    MJC_ENV_CLASS = BrittleStarLightEscapeMJCEnvironment
    MJX_ENV_CLASS = BrittleStarLightEscapeMJXEnvironment

    def __init__(
        self,
        env: (
            BrittleStarLightEscapeMJCEnvironment | BrittleStarLightEscapeMJXEnvironment
        ),
        backend: str,
    ) -> None:
        super().__init__(env=env, backend=backend)
