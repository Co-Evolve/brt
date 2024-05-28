from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np
from moojoco.environment.base import MuJoCoEnvironmentConfiguration
from moojoco.environment.mjc_env import MJCEnv, MJCEnvState, MJCObservable

from biorobot.brittle_star.environment.directed_locomotion.shared import (
    BrittleStarDirectedLocomotionEnvironmentConfiguration,
)
from biorobot.jumping_spider.environment.shared.base import JumpingSpiderEnvironmentBaseConfiguration, \
    JumpingSpiderEnvironmentBase
from biorobot.jumping_spider.environment.shared.mjc_observables import get_shared_jumping_spider_mjc_observables
from biorobot.jumping_spider.mjcf.arena.shared import MJCFSpiderArena
from biorobot.jumping_spider.mjcf.morphology.morphology import MJCFJumpingSpiderMorphology


class JumpingSpiderEnvironmentConfiguration(JumpingSpiderEnvironmentBaseConfiguration):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )


class JumpingSpiderMJCEnvironment(JumpingSpiderEnvironmentBase, MJCEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            mjcf_str: str,
            mjcf_assets: Dict[str, Any],
            configuration: MuJoCoEnvironmentConfiguration,
    ) -> None:
        JumpingSpiderEnvironmentBase.__init__(self)
        MJCEnv.__init__(
            self,
            mjcf_str=mjcf_str,
            mjcf_assets=mjcf_assets,
            configuration=configuration,
        )

    @property
    def environment_configuration(
            self,
    ) -> JumpingSpiderEnvironmentConfiguration:
        return super(MJCEnv, self).environment_configuration

    @classmethod
    def from_morphology_and_arena(
            cls,
            morphology: MJCFJumpingSpiderMorphology,
            arena: MJCFSpiderArena,
            configuration: BrittleStarDirectedLocomotionEnvironmentConfiguration,
    ) -> JumpingSpiderMJCEnvironment:
        return super().from_morphology_and_arena(morphology=morphology, arena=arena, configuration=configuration)

    def _get_mj_models_and_datas_to_render(
            self, state: MJCEnvState
    ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        mj_models, mj_datas = super()._get_mj_models_and_datas_to_render(state=state)
        if self.environment_configuration.color_contacts:
            self._color_segment_capsule_contacts(
                mj_models=mj_models, contact_bools=state.observations["leg_tip_contact"]
            )
        return mj_models, mj_datas

    def _create_observables(self) -> List[MJCObservable]:
        observables = get_shared_jumping_spider_mjc_observables(mj_model=self.frozen_mj_model,
                                                                mj_data=self.frozen_mj_data)
        # todo:
        #   - target platform pos / size (if target platform arena)
        return observables

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

        state = self._finish_reset(models_and_datas=(mj_model, mj_data), rng=rng)
        return state

    def _update_reward(self, state: MJCEnvState, previous_state: MJCEnvState) -> MJCEnvState:
        # todo: negative distance to target
        return state

    def _update_terminated(self, state: MJCEnvState) -> MJCEnvState:
        # todo: Target platform reached
        return state

    def _update_truncated(self, state: MJCEnvState) -> MJCEnvState:
        # todo:
        #   Time limit reached
        #   Z position below platform
        return state

    def _update_info(self, state: MJCEnvState) -> MJCEnvState:
        return state
