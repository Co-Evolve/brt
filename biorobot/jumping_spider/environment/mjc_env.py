from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from moojoco.environment.base import MuJoCoEnvironmentConfiguration
from moojoco.environment.mjc_env import MJCEnv, MJCEnvState, MJCObservable

from biorobot.brittle_star.environment.directed_locomotion.shared import (
    BrittleStarDirectedLocomotionEnvironmentConfiguration,
)
from biorobot.jumping_spider.mjcf.arena.shared import MJCFSpiderArena
from biorobot.jumping_spider.mjcf.morphology.morphology import MJCFJumpingSpiderMorphology


class JumpingSpiderEnvironmentConfiguration(MuJoCoEnvironmentConfiguration):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )


class JumpingSpiderMJCEnvironment(MJCEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            mjcf_str: str,
            mjcf_assets: Dict[str, Any],
            configuration: MuJoCoEnvironmentConfiguration,
    ) -> None:
        MJCEnv.__init__(
            self,
            mjcf_str=mjcf_str,
            mjcf_assets=mjcf_assets,
            configuration=configuration,
        )

    @property
    def environment_configuration(
            self,
    ) -> MuJoCoEnvironmentConfiguration:
        return super(MJCEnv, self).environment_configuration

    @classmethod
    def from_morphology_and_arena(
            cls,
            morphology: MJCFJumpingSpiderMorphology,
            arena: MJCFSpiderArena,
            configuration: BrittleStarDirectedLocomotionEnvironmentConfiguration,
    ) -> JumpingSpiderMJCEnvironment:
        return super().from_morphology_and_arena(morphology=morphology, arena=arena, configuration=configuration)

    def _create_observables(self) -> List[MJCObservable]:
        # todo:
        #   - torso pos
        #   - torso rotation
        #   - foot contacts
        #   - target platform pos / size (if target platform arena)
        #   - joint positions
        #   - joint velocities
        #   - actuator torques
        #   - joint torques
        #   - dragline length
        #   - force applied by tendon?
        return []

    @staticmethod
    def _get_time(state: MJCEnvState) -> float:
        return state.mj_data.time

    def reset(
            self,
            rng: np.random.RandomState,
            *args,
            **kwargs,
    ) -> MJCEnvState:
        mj_model, mj_data = self._prepare_reset()

        # Set morphology position
        mj_model.body("JumpingSpiderMorphology/cephalothorax").pos[2] = 0.5

        # Set arena dragline attachment
        arena_dragline_site = [site_id for site_id in range(mj_model.nsite) if
                               "dragline_attachment_site" in mj_model.site(site_id).name][0]
        spider_dragline_site = mj_data.site_xpos
        state = self._finish_reset(models_and_datas=(mj_model, mj_data), rng=rng)
        return state

    def _update_reward(self, state: MJCEnvState, previous_state: MJCEnvState) -> MJCEnvState:
        return state

    def _update_terminated(self, state: MJCEnvState) -> MJCEnvState:
        return state

    def _update_truncated(self, state: MJCEnvState) -> MJCEnvState:
        return state

    def _update_info(self, state: MJCEnvState) -> MJCEnvState:
        return state
