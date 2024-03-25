from __future__ import annotations

from typing import Any, Dict, List, Tuple

import chex
import jax.random
import mujoco
from jax import numpy as jnp
from moojoco.environment.mjx_env import MJXEnv, MJXEnvState, MJXObservable

from biorobot.brittle_star.environment.shared.mjx_observables import (
    get_shared_brittle_star_mjx_observables,
)
from biorobot.brittle_star.environment.undirected_locomotion.mjc_env import (
    BrittleStarUndirectedLocomotionEnvironmentConfiguration,
)
from biorobot.brittle_star.environment.undirected_locomotion.shared import (
    BrittleStarUndirectedLocomotionEnvironmentBase,
)
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology


class BrittleStarUndirectedLocomotionMJXEnvironment(
    BrittleStarUndirectedLocomotionEnvironmentBase, MJXEnv
):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        mjcf_str: str,
        mjcf_assets: Dict[str, Any],
        configuration: BrittleStarUndirectedLocomotionEnvironmentConfiguration,
    ) -> None:
        BrittleStarUndirectedLocomotionEnvironmentBase.__init__(self)
        MJXEnv.__init__(
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
        return super(MJXEnv, self).environment_configuration

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: MJCFBrittleStarMorphology,
        arena: MJCFAquariumArena,
        configuration: BrittleStarUndirectedLocomotionEnvironmentConfiguration,
    ) -> BrittleStarUndirectedLocomotionMJXEnvironment:
        return super().from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=configuration
        )

    def _create_observables(self) -> List[MJXObservable]:
        base_observables = get_shared_brittle_star_mjx_observables(
            mj_model=self.frozen_mj_model, mj_data=self.frozen_mj_data
        )
        return base_observables

    @staticmethod
    def _get_time(state: MJXEnvState) -> MJXEnvState:
        return state.mjx_data.time

    @staticmethod
    def _get_xy_distance_from_origin(state: MJXEnvState) -> float:
        disk_body_id = state.mj_model.body("BrittleStarMorphology/central_disk").id
        xy_disk_position = state.mjx_data.xpos[disk_body_id][:2]
        return jnp.linalg.norm(xy_disk_position)

    def _get_mj_models_and_datas_to_render(
        self, state: MJXEnvState
    ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        mj_models, mj_datas = super()._get_mj_models_and_datas_to_render(state=state)
        if self.environment_configuration.color_contacts:
            self._color_segment_capsule_contacts(
                mj_models=mj_models, contact_bools=state.observations["segment_contact"]
            )
        return mj_models, mj_datas

    def reset(self, rng: chex.PRNGKey, *args, **kwargs) -> MJXEnvState:
        (mj_model, mj_data), (mjx_model, mjx_data) = self._prepare_reset()

        rng, qpos_rng, qvel_rng = jax.random.split(key=rng, num=3)

        disk_body_id = mj_model.body("BrittleStarMorphology/central_disk").id

        # Set morphology position
        morphology_pos = jnp.array([0.0, 0.0, 0.11])
        mjx_model = mjx_model.replace(
            body_pos=mjx_model.body_pos.at[disk_body_id].set(morphology_pos)
        )

        # Add noise to initial qpos and qvel of segment joints
        qpos = jnp.copy(mjx_model.qpos0)
        qvel = jnp.zeros(mjx_model.nv)

        joint_qpos_adrs = self._get_segment_joints_qpos_adrs(mj_model=mj_model)
        joint_qvel_adrs = self._get_segment_joints_qvel_adrs(mj_model=mj_model)
        num_segment_joints = len(joint_qpos_adrs)

        qpos = qpos.at[joint_qpos_adrs].set(
            qpos[joint_qpos_adrs]
            + jax.random.uniform(
                key=qpos_rng,
                shape=(num_segment_joints,),
                minval=-self.environment_configuration.joint_randomization_noise_scale,
                maxval=self.environment_configuration.joint_randomization_noise_scale,
            )
        )
        qvel = qvel.at[joint_qvel_adrs].set(
            jax.random.uniform(
                key=qvel_rng,
                shape=(num_segment_joints,),
                minval=-self.environment_configuration.joint_randomization_noise_scale,
                maxval=self.environment_configuration.joint_randomization_noise_scale,
            )
        )

        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

        state = self._finish_reset(
            models_and_datas=((mj_model, mj_data), (mjx_model, mjx_data)), rng=rng
        )
        return state
