from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np
from moojoco.environment.mjc_env import MJCEnv, MJCEnvState, MJCObservable

from biorobot.brittle_star.environment.shared.mjc_observables import (
    get_shared_brittle_star_mjc_observables,
)
from biorobot.brittle_star.environment.undirected_locomotion.shared import (
    BrittleStarUndirectedLocomotionEnvironmentBase,
    BrittleStarUndirectedLocomotionEnvironmentConfiguration,
)
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology


class BrittleStarUndirectedLocomotionMJCEnvironment(
    BrittleStarUndirectedLocomotionEnvironmentBase, MJCEnv
):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        mjcf_str: str,
        mjcf_assets: Dict[str, Any],
        configuration: BrittleStarUndirectedLocomotionEnvironmentConfiguration,
    ) -> None:
        BrittleStarUndirectedLocomotionEnvironmentBase.__init__(self)
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
    ) -> BrittleStarUndirectedLocomotionEnvironmentConfiguration:
        return super(MJCEnv, self).environment_configuration

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: MJCFBrittleStarMorphology,
        arena: MJCFAquariumArena,
        configuration: BrittleStarUndirectedLocomotionEnvironmentConfiguration,
    ) -> BrittleStarUndirectedLocomotionMJCEnvironment:
        return super().from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=configuration
        )

    def _create_observables(self) -> List[MJCObservable]:
        base_observables = get_shared_brittle_star_mjc_observables(
            mj_model=self.frozen_mj_model, mj_data=self.frozen_mj_data
        )
        return base_observables

    @staticmethod
    def _get_time(state: MJCEnvState) -> MJCEnvState:
        return state.mj_data.time

    @staticmethod
    def _get_xy_distance_from_origin(state: MJCEnvState) -> float:
        disk_body_id = state.mj_model.body("BrittleStarMorphology/central_disk").id
        xy_disk_position = state.mj_data.body(disk_body_id).xpos[:2]
        return np.linalg.norm(xy_disk_position)

    def _get_mj_models_and_datas_to_render(
        self, state: MJCEnvState
    ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        mj_models, mj_datas = super()._get_mj_models_and_datas_to_render(state=state)
        if self.environment_configuration.color_contacts:
            self._color_segment_capsule_contacts(
                mj_models=mj_models, contact_bools=state.observations["segment_contact"]
            )
        return mj_models, mj_datas

    def reset(self, rng: np.random.RandomState, *args, **kwargs) -> MJCEnvState:
        mj_model, mj_data = self._prepare_reset()

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
