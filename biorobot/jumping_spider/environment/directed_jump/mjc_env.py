from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np
from moojoco.environment.base import MuJoCoEnvironmentConfiguration
from moojoco.environment.mjc_env import MJCEnv, MJCEnvState, MJCObservable

from biorobot.jumping_spider.environment.shared.base import JumpingSpiderEnvironmentBaseConfiguration, \
    JumpingSpiderEnvironmentBase
from biorobot.jumping_spider.environment.shared.mjc_observables import get_shared_jumping_spider_mjc_observables
from biorobot.jumping_spider.mjcf.arena.directed_jump import MJCFDirectedJumpArena
from biorobot.jumping_spider.mjcf.morphology.morphology import MJCFJumpingSpiderMorphology


class JumpingSpiderDirectedJumpEnvironmentConfiguration(JumpingSpiderEnvironmentBaseConfiguration):
    def __init__(
            self,
            target_distance_range: Tuple[float, float],
            target_angle_range: Tuple[float, float],
            *args,
            **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.target_distance_range = target_distance_range
        self.target_angle_range = target_angle_range


class JumpingSpiderDirectedJumpMJCEnvironment(JumpingSpiderEnvironmentBase, MJCEnv):
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
    ) -> JumpingSpiderDirectedJumpEnvironmentConfiguration:
        return super(MJCEnv, self).environment_configuration

    @classmethod
    def from_morphology_and_arena(
            cls,
            morphology: MJCFJumpingSpiderMorphology,
            arena: MJCFDirectedJumpArena,
            configuration: JumpingSpiderDirectedJumpEnvironmentConfiguration,
    ) -> JumpingSpiderDirectedJumpMJCEnvironment:
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
    def _get_xy_direction_to_target(state: MJCEnvState) -> np.ndarray:
        target_position = state.mj_data.body("target").xpos
        disk_position = state.mj_data.body("BrittleStarMorphology/central_disk").xpos
        direction_to_target = target_position - disk_position
        return direction_to_target[:2]

    @staticmethod
    def _get_target_position(state: MJCEnvState) -> np.ndarray:
        return state.mj_data.body("target").xpos

    @staticmethod
    def _get_spider_position(state: MJCEnvState) -> np.ndarray:
        return state.mj_data.body("JumpingSpiderMorphology/cephalothorax").xpos

    @staticmethod
    def _get_direction_to_target(state: MJCEnvState) -> np.ndarray:
        return JumpingSpiderDirectedJumpMJCEnvironment._get_target_position(
            state=state) - JumpingSpiderDirectedJumpMJCEnvironment._get_spider_position(state=state)

    @staticmethod
    def _get_distance_to_target(state: MJCEnvState) -> np.ndarray:
        direction_to_target = JumpingSpiderDirectedJumpMJCEnvironment._get_direction_to_target(state=state)
        distance_to_target = np.linalg.norm(direction_to_target)
        return distance_to_target

    def _create_observables(self) -> List[MJCObservable]:
        observables = get_shared_jumping_spider_mjc_observables(mj_model=self.frozen_mj_model,
                                                                mj_data=self.frozen_mj_data)

        direction_to_target = MJCObservable(
            name="unit_direction_to_target",
            low=-np.ones(3),
            high=np.ones(3),
            retriever=lambda state: self._get_direction_to_target(
                state=state) / self._get_distance_to_target(
                state=state)
        )

        distance_to_target = MJCObservable(
            name="distance_to_target",
            low=np.zeros(1),
            high=np.inf * np.ones(1),
            retriever=lambda state: np.array([self._get_distance_to_target(state=state)])
        )
        return observables + [direction_to_target, distance_to_target]

    @staticmethod
    def _get_time(state: MJCEnvState) -> float:
        return state.mj_data.time

    def _generate_target_position(self, rng: np.random.RandomState,
                                  target_position: np.ndarray | None) -> np.ndarray:
        if target_position is not None:
            position = np.array(target_position)
        else:
            distance = rng.uniform(self.environment_configuration.target_distance_range[0],
                                   self.environment_configuration.target_distance_range[1])
            angle = rng.uniform(self.environment_configuration.target_angle_range[0],
                                self.environment_configuration.target_angle_range[1])
            position = distance * np.array([np.cos(angle), 0, np.sin(angle)])

        return position

    def reset(
            self,
            rng: np.random.RandomState,
            target_position: np.ndarray | None = None,
            *args,
            **kwargs,
    ) -> MJCEnvState:
        mj_model, mj_data = self._prepare_reset()

        # Set morphology position
        mj_model.body("JumpingSpiderMorphology/cephalothorax").pos[2] = 1

        # Set random target position
        mj_model.body("target").pos = self._generate_target_position(
            rng=rng, target_position=target_position
        )

        state = self._finish_reset(models_and_datas=(mj_model, mj_data), rng=rng)
        return state

    def _update_reward(self, state: MJCEnvState, previous_state: MJCEnvState) -> MJCEnvState:
        current_distance_to_target = self._get_distance_to_target(state=state)
        previous_distance_to_target = self._get_distance_to_target(state=state)

        reward = previous_distance_to_target - current_distance_to_target

        # noinspection PyUnresolvedReferences
        return state.replace(reward=reward)

    def _update_terminated(self, state: MJCEnvState) -> MJCEnvState:
        terminated = self._get_distance_to_target(state=state) < 0.2

        # noinspection PyUnresolvedReferences
        return state.replace(terminated=terminated)

    def _update_truncated(self, state: MJCEnvState) -> MJCEnvState:
        truncated = self._get_time(state=state) > self.environment_configuration.simulation_time

        # noinspection PyUnresolvedReferences
        return state.replace(truncated=truncated)

    def _update_info(self, state: MJCEnvState) -> MJCEnvState:
        info = {
            "time": self._get_time(state=state),
            "target_position": self._get_target_position(state=state)
        }

        # noinspection PyUnresolvedReferences
        return state.replace(info=info)
