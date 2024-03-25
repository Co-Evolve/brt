from __future__ import annotations

from typing import Any, Dict, List, Tuple

import chex
import jax.random
import mujoco
from jax import numpy as jnp
from moojoco.environment.mjx_env import MJXEnv, MJXEnvState, MJXObservable

from biorobot.brittle_star.environment.directed_locomotion.shared import (
    BrittleStarDirectedLocomotionEnvironmentBase,
    BrittleStarDirectedLocomotionEnvironmentConfiguration,
)
from biorobot.brittle_star.environment.shared.mjx_observables import (
    get_shared_brittle_star_mjx_observables,
)
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology


class BrittleStarDirectedLocomotionMJXEnvironment(
    BrittleStarDirectedLocomotionEnvironmentBase, MJXEnv
):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        mjcf_str: str,
        mjcf_assets: Dict[str, Any],
        configuration: BrittleStarDirectedLocomotionEnvironmentConfiguration,
    ) -> None:
        BrittleStarDirectedLocomotionEnvironmentBase.__init__(self)
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
    ) -> BrittleStarDirectedLocomotionEnvironmentConfiguration:
        return super(MJXEnv, self).environment_configuration

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: MJCFBrittleStarMorphology,
        arena: MJCFAquariumArena,
        configuration: BrittleStarDirectedLocomotionEnvironmentConfiguration,
    ) -> BrittleStarDirectedLocomotionMJXEnvironment:
        assert arena.arena_configuration.attach_target, (
            f"Arena must have a target attached. Please set "
            f"'attach_target' to 'True' in the "
            f"AquariumArenaConfiguration."
        )
        return super().from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=configuration
        )

    @staticmethod
    def _get_xy_direction_to_target(state: MJXEnvState) -> jnp.ndarray:
        disk_body_id = state.mj_model.body("BrittleStarMorphology/central_disk").id
        target_body_id = state.mj_model.body("target").id

        disk_position = state.mjx_data.xpos[disk_body_id]
        target_position = state.mjx_data.xpos[target_body_id]
        direction_to_target = target_position - disk_position
        return direction_to_target[:2]

    @staticmethod
    def _get_xy_distance_to_target(state: MJXEnvState) -> float:
        xy_direction_to_target = (
            BrittleStarDirectedLocomotionMJXEnvironment._get_xy_direction_to_target(
                state=state
            )
        )
        xy_distance_to_target = jnp.linalg.norm(xy_direction_to_target)
        return xy_distance_to_target

    @staticmethod
    def _get_xy_target_position(state: MJXEnvState) -> jnp.ndarray:
        target_body_id = state.mj_model.body("target").id
        return state.mjx_data.xpos[target_body_id][:2]

    def _create_observables(self) -> List[MJXObservable]:
        base_observables = get_shared_brittle_star_mjx_observables(
            mj_model=self.frozen_mj_model, mj_data=self.frozen_mj_data
        )

        unit_xy_direction_to_target_observable = MJXObservable(
            name="unit_xy_direction_to_target",
            low=-jnp.ones(2),
            high=jnp.ones(2),
            retriever=lambda state: self._get_xy_direction_to_target(state=state)
            / self._get_xy_distance_to_target(state=state),
        )
        # distance to target
        xy_distance_to_target_observable = MJXObservable(
            name="xy_distance_to_target",
            low=jnp.zeros(1),
            high=jnp.inf * jnp.ones(1),
            retriever=lambda state: jnp.array([self._get_xy_distance_to_target(state)]),
        )

        return base_observables + [
            unit_xy_direction_to_target_observable,
            xy_distance_to_target_observable,
        ]

    @staticmethod
    def _get_time(state: MJXEnvState) -> float:
        return state.mjx_data.time

    def _get_mj_models_and_datas_to_render(
        self, state: MJXEnvState
    ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        mj_models, mj_datas = super()._get_mj_models_and_datas_to_render(state=state)
        if self.environment_configuration.color_contacts:
            self._color_segment_capsule_contacts(
                mj_models=mj_models, contact_bools=state.observations["segment_contact"]
            )
        return mj_models, mj_datas

    def _get_target_position(
        self, rng: jnp.ndarray, target_position: jnp.ndarray
    ) -> jnp.ndarray:
        def return_given_target_position() -> jnp.ndarray:
            return target_position

        def return_random_target_position() -> jnp.ndarray:
            angle = jax.random.uniform(key=rng, shape=(), minval=0, maxval=jnp.pi * 2)
            radius = self.environment_configuration.target_distance
            random_position = jnp.array(
                [radius * jnp.cos(angle), radius * jnp.sin(angle), 0.05]
            )
            return random_position

        return jax.lax.cond(
            jnp.any(jnp.isnan(target_position)),
            return_random_target_position,
            return_given_target_position,
        )

    def reset(
        self,
        rng: chex.PRNGKey,
        target_position: jnp.ndarray = jnp.array([jnp.nan, jnp.nan, jnp.nan]),
        *args,
        **kwargs,
    ) -> MJXEnvState:
        (mj_model, mj_data), (mjx_model, mjx_data) = self._prepare_reset()

        rng, target_pos_rng, qpos_rng, qvel_rng = jax.random.split(key=rng, num=4)

        target_body_id = mj_model.body("target").id
        disk_body_id = mj_model.body("BrittleStarMorphology/central_disk").id

        # Set random target position
        target_pos = self._get_target_position(
            rng=target_pos_rng, target_position=jnp.array(target_position)
        )
        mjx_model = mjx_model.replace(
            body_pos=mjx_model.body_pos.at[target_body_id].set(target_pos)
        )

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
