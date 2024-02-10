from moojoco.environment.dual import DualMuJoCoEnvironment

from biorobot.brittle_star.environment.undirected_locomotion.mjc_env import (
    BrittleStarUndirectedLocomotionMJCEnvironment,
)
from biorobot.brittle_star.environment.undirected_locomotion.mjx_env import (
    BrittleStarUndirectedLocomotionMJXEnvironment,
)


class BrittleStarUndirectedLocomotionEnvironment(DualMuJoCoEnvironment):
    MJC_ENV_CLASS = BrittleStarUndirectedLocomotionMJCEnvironment
    MJX_ENV_CLASS = BrittleStarUndirectedLocomotionMJXEnvironment

    def __init__(
        self,
        env: (
            BrittleStarUndirectedLocomotionMJCEnvironment
            | BrittleStarUndirectedLocomotionMJXEnvironment
        ),
        backend: str,
    ) -> None:
        super().__init__(env=env, backend=backend)
