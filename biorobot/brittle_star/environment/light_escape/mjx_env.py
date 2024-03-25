from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import chex
import jax.random
import mujoco
import numpy as np
from gymnasium.core import RenderFrame
from jax import numpy as jnp
from moojoco.environment.mjx_env import MJXEnv, MJXEnvState, MJXObservable
from moojoco.environment.renderer import MujocoRenderer
from mujoco import mjx

from biorobot.brittle_star.environment.light_escape.shared import (
    BrittleStarLightEscapeEnvironmentBase,
    BrittleStarLightEscapeEnvironmentConfiguration,
)
from biorobot.brittle_star.environment.shared.mjx_observables import (
    get_shared_brittle_star_mjx_observables,
)
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.utils.noise import generate_perlin_noise_2d


class BrittleStarLightEscapeMJXEnvironment(
    BrittleStarLightEscapeEnvironmentBase, MJXEnv
):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        mjcf_str: str,
        mjcf_assets: Dict[str, Any],
        configuration: BrittleStarLightEscapeEnvironmentConfiguration,
    ) -> None:
        BrittleStarLightEscapeEnvironmentBase.__init__(self)
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
    ) -> BrittleStarLightEscapeEnvironmentConfiguration:
        return super(MJXEnv, self).environment_configuration

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: MJCFBrittleStarMorphology,
        arena: MJCFAquariumArena,
        configuration: BrittleStarLightEscapeEnvironmentConfiguration,
    ) -> BrittleStarLightEscapeMJXEnvironment:
        assert arena.arena_configuration.sand_ground_color, (
            "This environment requires the 'sand_ground_color' "
            "parameter of the AquariumArenaConfiguration to be "
            "set to 'True'."
        )
        return super().from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=configuration
        )

    @staticmethod
    def _get_x_distance_from_start_position(state: MJXEnvState) -> float:
        disk_body_id = state.mj_model.body("BrittleStarMorphology/central_disk").id
        x_disk_position = state.mjx_data.xpos[disk_body_id][0]
        start_x_position = BrittleStarLightEscapeEnvironmentBase._get_x_start_position(
            mj_model=state.mj_model
        )
        return x_disk_position - start_x_position

    def _get_light_per_segment(self, state: MJXEnvState) -> jnp.ndarray:
        arena_size = jnp.array(state.mj_model.geom("groundplane").size[:2])
        light_map_size = jnp.array(state.info["_light_map"].shape)

        def _get_light_of_segment(geom_id: int) -> float:
            segment_xy_position = state.mjx_data.geom_xpos[geom_id, :2]
            shifted_segment_xy_position = segment_xy_position + arena_size
            normalized_segment_xy_position = shifted_segment_xy_position / (
                2 * arena_size
            )
            normalized_segment_xy_position = normalized_segment_xy_position.at[1].set(
                1 - normalized_segment_xy_position[1]
            )
            segment_yx_light_map_position = (
                normalized_segment_xy_position[::-1] * light_map_size
            )

            light_map_coords = jnp.round(segment_yx_light_map_position).astype(int)
            y_coord = jnp.clip(a=light_map_coords[0], a_min=0, a_max=light_map_size[0])
            x_coord = jnp.clip(a=light_map_coords[1], a_min=0, a_max=light_map_size[1])
            return state.info["_light_map"][y_coord, x_coord]

        return jax.vmap(_get_light_of_segment)(
            jnp.array(self._get_segment_capsule_geom_ids(mj_model=state.mj_model))
        )

    def _get_average_light_income(self, state: MJXEnvState) -> jnp.ndarray:
        return jnp.average(self._get_light_per_segment(state=state))

    def _create_observables(self) -> List[MJXObservable]:
        base_observables = get_shared_brittle_star_mjx_observables(
            mj_model=self.frozen_mj_model, mj_data=self.frozen_mj_data
        )

        num_segments = len(
            self._get_segment_capsule_geom_ids(mj_model=self.frozen_mj_model)
        )
        segment_light_intake_observable = MJXObservable(
            name="segment_light_intake",
            low=np.zeros(num_segments),
            high=np.ones(num_segments),
            retriever=lambda state: self._get_light_per_segment(state=state),
        )
        return base_observables + [segment_light_intake_observable]

    @staticmethod
    def _get_time(state: MJXEnvState) -> float:
        return state.mjx_data.time

    def _get_mj_models_and_datas_to_render(
        self, state: MJXEnvState
    ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        mj_models, mj_datas = super()._get_mj_models_and_datas_to_render(state=state)
        self._update_mj_models_tex_rgb(mj_models=mj_models, state=state)
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

    def get_renderer_context(
        self, renderer: Union[MujocoRenderer, mujoco.Renderer]
    ) -> mujoco.MjrContext:
        return super(MJXEnv, self).get_renderer_context(renderer)

    def render(self, state: MJXEnvState) -> list[RenderFrame] | None:
        render_output = super().render(state=state)
        state.info["_light_map_has_changed"] = (
            state.info["_light_map_has_changed"] * False
        )
        return render_output

    def _update_light_map(self, state: MJXEnvState) -> MJXEnvState:
        rng, sub_rng = jax.random.split(state.rng)

        ground_texture = state.mj_model.texture("groundplane")
        h, w = ground_texture.height[0], ground_texture.width[0]

        # Linear transition from bright to dark
        light_map = jnp.ones((h, w))
        light_map = light_map.at[:].set(jnp.linspace(1, 0, w))

        if self.environment_configuration.light_perlin_noise_scale > 0:
            starting_zone_width = w // 5

            # generate perlin noise
            res = (self.environment_configuration.light_perlin_noise_scale,) * 2
            light_noise = generate_perlin_noise_2d(
                shape=light_map.shape, res=res, rng_key=sub_rng, npi=jnp
            )  # Normalize light map
            light_noise = 1 - 2 * light_noise

            light_noise_weights = jnp.zeros_like(light_noise)
            light_noise_weights = light_noise_weights.at[:, starting_zone_width:].set(
                jnp.linspace(0, 1, w - starting_zone_width)
            )

            light_noise = light_noise * light_noise_weights

            light_noise = 1 + light_noise
            light_map = light_map * light_noise

            light_map = (light_map - jnp.min(light_map)) / (
                jnp.max(light_map) - jnp.min(light_map)
            )

        info = state.info
        info.update({"_light_map": light_map, "_light_map_has_changed": True})
        # noinspection PyUnresolvedReferences
        return state.replace(rng=rng, info=info)

    def _finish_reset(
        self,
        models_and_datas: Tuple[
            Tuple[mujoco.MjModel, mujoco.MjData], Tuple[mjx.Model, mjx.Data]
        ],
        rng: np.random.RandomState,
    ) -> MJXEnvState:
        (mj_model, mj_data), (mjx_model, mjx_data) = models_and_datas
        mjx_data = mjx.forward(m=mjx_model, d=mjx_data)
        # noinspection PyArgumentList
        state = MJXEnvState(
            mj_model=mj_model,
            mj_data=mj_data,
            mjx_model=mjx_model,
            mjx_data=mjx_data,
            observations={},
            reward=0,
            terminated=False,
            truncated=False,
            info={},
            rng=rng,
        )
        state = self._update_info(state=state)
        state = self._update_light_map(state=state)
        state = self._update_observations(state=state)
        return state

    def reset(self, rng: chex.PRNGKey, *args, **kwargs) -> MJXEnvState:
        (mj_model, mj_data), (mjx_model, mjx_data) = self._prepare_reset()

        rng, qpos_rng, qvel_rng = jax.random.split(key=rng, num=3)

        # Set morphology position
        disk_body_id = mj_model.body("BrittleStarMorphology/central_disk").id
        morphology_pos = jnp.array(
            [
                BrittleStarLightEscapeEnvironmentBase._get_x_start_position(
                    mj_model=mj_model
                ),
                0.0,
                0.11,
            ]
        )
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
