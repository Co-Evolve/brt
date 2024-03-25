from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np
from moojoco.environment.mjc_env import MJCEnv, MJCEnvState, MJCObservable

from biorobot.brittle_star.environment.directed_locomotion.shared import (
    BrittleStarDirectedLocomotionEnvironmentBase,
    BrittleStarDirectedLocomotionEnvironmentConfiguration,
)
from biorobot.brittle_star.environment.shared.mjc_observables import (
    get_shared_brittle_star_mjc_observables,
)
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology


class BrittleStarDirectedLocomotionMJCEnvironment(
    BrittleStarDirectedLocomotionEnvironmentBase, MJCEnv
):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        mjcf_str: str,
        mjcf_assets: Dict[str, Any],
        configuration: BrittleStarDirectedLocomotionEnvironmentConfiguration,
    ) -> None:
        BrittleStarDirectedLocomotionEnvironmentBase.__init__(self)
        MJCEnv.__init__(
            self,
            mjcf_str=mjcf_str,
            mjcf_assets=mjcf_assets,
            configuration=configuration,
        )
        self._cache_references(mj_model=self.frozen_mj_model)

    @property
    def environment_configuration(
        self,
    ) -> BrittleStarDirectedLocomotionEnvironmentConfiguration:
        return super(MJCEnv, self).environment_configuration

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: MJCFBrittleStarMorphology,
        arena: MJCFAquariumArena,
        configuration: BrittleStarDirectedLocomotionEnvironmentConfiguration,
    ) -> BrittleStarDirectedLocomotionMJCEnvironment:
        assert arena.arena_configuration.attach_target, (
            f"Arena must have a target attached. Please set "
            f"'attach_target' to 'True' in the "
            f"AquariumArenaConfiguration."
        )
        return super().from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=configuration
        )

    @staticmethod
    def _get_xy_direction_to_target(state: MJCEnvState) -> np.ndarray:
        target_position = state.mj_data.body("target").xpos
        disk_position = state.mj_data.body("BrittleStarMorphology/central_disk").xpos
        direction_to_target = target_position - disk_position
        return direction_to_target[:2]

    @staticmethod
    def _get_xy_distance_to_target(state: MJCEnvState) -> float:
        xy_direction_to_target = (
            BrittleStarDirectedLocomotionMJCEnvironment._get_xy_direction_to_target(
                state=state
            )
        )
        xy_distance_to_target = np.linalg.norm(xy_direction_to_target)
        return xy_distance_to_target

    @staticmethod
    def _get_xy_target_position(state: MJCEnvState) -> np.ndarray:
        return np.array(state.mj_data.body("target").xpos[:2])

    def _create_observables(self) -> List[MJCObservable]:
        base_observables = get_shared_brittle_star_mjc_observables(
            mj_model=self.frozen_mj_model, mj_data=self.frozen_mj_data
        )

        # direction to target
        unit_xy_direction_to_target_observable = MJCObservable(
            name="unit_xy_direction_to_target",
            low=-np.ones(2),
            high=np.ones(2),
            retriever=lambda state: self._get_xy_direction_to_target(state)
            / self._get_xy_distance_to_target(state=state),
        )

        # distance to target
        xy_distance_to_target_observable = MJCObservable(
            name="xy_distance_to_target",
            low=np.zeros(1),
            high=np.inf * np.ones(1),
            retriever=lambda state: np.array(
                [self._get_xy_distance_to_target(state=state)]
            ),
        )

        return base_observables + [
            unit_xy_direction_to_target_observable,
            xy_distance_to_target_observable,
        ]

    @staticmethod
    def _get_time(state: MJCEnvState) -> float:
        return state.mj_data.time

    def _get_mj_models_and_datas_to_render(
        self, state: MJCEnvState
    ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        mj_models, mj_datas = super()._get_mj_models_and_datas_to_render(state=state)
        if self.environment_configuration.color_contacts:
            self._color_segment_capsule_contacts(
                mj_models=mj_models, contact_bools=state.observations["segment_contact"]
            )
        return mj_models, mj_datas

    def _get_target_position(
        self, rng: np.random.RandomState, target_position: np.ndarray | None = None
    ) -> np.ndarray:
        if target_position is not None:
            position = np.array(target_position)
        else:
            angle = rng.uniform(0, 2 * np.pi)
            radius = self.environment_configuration.target_distance
            position = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.05])
        return position

    def reset(
        self,
        rng: np.random.RandomState,
        target_position: np.ndarray | None = None,
        *args,
        **kwargs,
    ) -> MJCEnvState:
        mj_model, mj_data = self._prepare_reset()

        # Set random target position
        mj_model.body("target").pos = self._get_target_position(
            rng=rng, target_position=target_position
        )

        # Set morphology position
        mj_model.body("BrittleStarMorphology/central_disk").pos[2] = 0.11

        # Add noise to initial qpos and qvel of segment joints
        joint_qpos_adrs = self._get_segment_joints_qpos_adrs(mj_model=mj_model)
        joint_qvel_adrs = self._get_segment_joints_qvel_adrs(mj_model=mj_model)
        num_segment_joints = len(joint_qpos_adrs)

        mj_data.qpos[joint_qpos_adrs] = mj_model.qpos0[joint_qpos_adrs] + rng.uniform(
            low=-self.environment_configuration.joint_randomization_noise_scale,
            high=self.environment_configuration.joint_randomization_noise_scale,
            size=num_segment_joints,
        )
        mj_data.qvel[joint_qvel_adrs] = rng.uniform(
            low=-self.environment_configuration.joint_randomization_noise_scale,
            high=self.environment_configuration.joint_randomization_noise_scale,
            size=num_segment_joints,
        )

        state = self._finish_reset(models_and_datas=(mj_model, mj_data), rng=rng)
        return state
