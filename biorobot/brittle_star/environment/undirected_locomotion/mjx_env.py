from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import chex
import jax.random
import mujoco
from gymnasium.core import RenderFrame
from jax import numpy as jnp
from moojoco.environment.mjx_env import MJXEnv, MJXEnvState, MJXObservable
from moojoco.environment.renderer import MujocoRenderer

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
from biorobot.utils.heightmap import generate_hfield, generate_radial_matrix


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

    def get_renderer(
            self,
            identifier: int,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: MJXEnvState,
    ) -> Union[MujocoRenderer, mujoco.Renderer]:
        renderer = super().get_renderer(
            identifier=identifier, model=model, data=data, state=state
        )
        self._update_renderer_context(mj_model=model, state=state, renderer=renderer)
        return renderer

    def render(self, state: MJXEnvState) -> list[RenderFrame] | None:
        render_output = super().render(state=state)
        state.info["_hfield_has_changed"] = (
                state.info["_hfield_has_changed"] * False
        )
        return render_output

    def _update_hfield(self, state: MJXEnvState) -> MJXEnvState:
        if self.environment_configuration.hfield_perlin_noise_scale > 0:
            hfield = state.mj_model.hfield("groundplane_hfield")

            rng, hfield_rng = jax.random.split(key=state.rng, num=2)
            hfield_data = generate_hfield(shape=hfield.data.shape, rng=hfield_rng,
                                          noise_scale=self.environment_configuration.hfield_perlin_noise_scale, npi=jnp)

            mask = generate_radial_matrix(shape=hfield_data.shape, inner_radius=5, outer_radius=100, npi=jnp)
            hfield_data = hfield_data * mask

            mjx_model = state.mjx_model.replace(hfield_data=hfield_data.flatten())

            # noinspection PyUnresolvedReferences
            state = state.replace(
                mjx_model=mjx_model,
                rng=rng
            )
            state.info.update({"_hfield_has_changed": True})
        else:
            state.info.update({"_hfield_has_changed": False})

        return state

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
        state = self._update_hfield(state=state)

        return state
