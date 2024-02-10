from moojoco.environment.dual import DualMuJoCoEnvironment

from biorobot.brittle_star.environment.directed_locomotion.mjc_env import (
    BrittleStarDirectedLocomotionMJCEnvironment,
)
from biorobot.brittle_star.environment.directed_locomotion.mjx_env import (
    BrittleStarDirectedLocomotionMJXEnvironment,
)


class BrittleStarDirectedLocomotionEnvironment(DualMuJoCoEnvironment):
    MJC_ENV_CLASS = BrittleStarDirectedLocomotionMJCEnvironment
    MJX_ENV_CLASS = BrittleStarDirectedLocomotionMJXEnvironment

    def __init__(
        self,
        env: (
            BrittleStarDirectedLocomotionMJCEnvironment
            | BrittleStarDirectedLocomotionMJXEnvironment
        ),
        backend: str,
    ) -> None:
        super().__init__(env=env, backend=backend)
