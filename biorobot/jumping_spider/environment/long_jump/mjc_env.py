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
from biorobot.jumping_spider.mjcf.arena.long_jump import MJCFLongJumpArena
from biorobot.jumping_spider.mjcf.morphology.morphology import MJCFJumpingSpiderMorphology


class JumpingSpiderLongJumpEnvironmentConfiguration(JumpingSpiderEnvironmentBaseConfiguration):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )


class JumpingSpiderLongJumpMJCEnvironment(JumpingSpiderEnvironmentBase, MJCEnv):
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
    ) -> JumpingSpiderLongJumpEnvironmentConfiguration:
        return super(MJCEnv, self).environment_configuration

    @classmethod
    def from_morphology_and_arena(
            cls,
            morphology: MJCFJumpingSpiderMorphology,
            arena: MJCFLongJumpArena,
            configuration: BrittleStarDirectedLocomotionEnvironmentConfiguration,
    ) -> JumpingSpiderLongJumpMJCEnvironment:
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

    @staticmethod
    def _get_spider_position(state: MJCEnvState) -> np.ndarray:
        return state.mj_data.body("JumpingSpiderMorphology/cephalothorax").xpos

    @staticmethod
    def _get_x_distance_from_origin(state: MJCEnvState) -> np.ndarray:
        spider_x_position = JumpingSpiderLongJumpMJCEnvironment._get_spider_position(state=state)[0]
        return spider_x_position

    @staticmethod
    def _get_track_width(state: MJCEnvState) -> float:
        return state.mj_model.geom("LongJumpArena_track_geom").size[1]

    def _create_observables(self) -> List[MJCObservable]:
        observables = get_shared_jumping_spider_mjc_observables(mj_model=self.frozen_mj_model,
                                                                mj_data=self.frozen_mj_data)

        x_distance = MJCObservable(
            name="x_distance_from_origin",
            low=-np.inf * np.ones(1),
            high=np.inf * np.ones(1),
            retriever=lambda state: self._get_x_distance_from_origin(state=state)
        )

        return observables + [x_distance]

    @staticmethod
    def _get_time(state: MJCEnvState) -> float:
        return state.mj_data.time

    def reset(
            self,
            rng: np.random.RandomState,
            target_platform_position: np.ndarray | None = None,
            *args,
            **kwargs,
    ) -> MJCEnvState:
        mj_model, mj_data = self._prepare_reset()

        # Set morphology position
        mj_model.body("JumpingSpiderMorphology/cephalothorax").pos[2] = 0.5

        state = self._finish_reset(models_and_datas=(mj_model, mj_data), rng=rng)
        return state

    def _update_reward(self, state: MJCEnvState, previous_state: MJCEnvState) -> MJCEnvState:
        # reward based on distance travelled from origin
        previous_x_distance_from_origin = self._get_x_distance_from_origin(state=previous_state)
        current_x_distance_from_origin = self._get_x_distance_from_origin(state=state)
        reward = current_x_distance_from_origin - previous_x_distance_from_origin

        # noinspection PyUnresolvedReferences
        return state.replace(reward=reward)

    def _update_terminated(self, state: MJCEnvState) -> MJCEnvState:
        # If x distance > 1 and height is low
        terminated = self._get_x_distance_from_origin(state=state) > 1
        terminated &= self._get_spider_position(state=state)[2] < 1

        # noinspection PyUnresolvedReferences
        return state.replace(terminated=terminated)

    def _update_truncated(self, state: MJCEnvState) -> MJCEnvState:
        truncated = self._get_spider_position(state=state)[1] > self._get_track_width(state=state)
        truncated |= self._get_time(state=state) > self.environment_configuration.simulation_time

        # noinspection PyUnresolvedReferences
        return state.replace(truncated=truncated)

    def _update_info(self, state: MJCEnvState) -> MJCEnvState:
        info = {
            "time": self._get_time(state=state),
        }

        # noinspection PyUnresolvedReferences
        return state.replace(info=info)
