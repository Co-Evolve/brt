from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import mujoco
import numpy as np
from gymnasium.core import RenderFrame
from moojoco.environment.mjc_env import MJCEnv, MJCEnvState, MJCObservable
from moojoco.environment.renderer import MujocoRenderer
from transforms3d.euler import euler2quat

from biorobot.brittle_star.environment.light_escape.shared import (
    BrittleStarLightEscapeEnvironmentBase,
    BrittleStarLightEscapeEnvironmentConfiguration,
)
from biorobot.brittle_star.environment.shared.observables import (
    get_base_brittle_star_observables,
)
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.utils.noise import generate_perlin_noise_2d


class BrittleStarLightEscapeMJCEnvironment(
    BrittleStarLightEscapeEnvironmentBase, MJCEnv
):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        mjcf_str: str,
        mjcf_assets: Dict[str, Any],
        configuration: BrittleStarLightEscapeEnvironmentConfiguration,
    ) -> None:
        BrittleStarLightEscapeEnvironmentBase.__init__(self)
        MJCEnv.__init__(
            self,
            mjcf_str=mjcf_str,
            mjcf_assets=mjcf_assets,
            configuration=configuration,
        )
        self._cache_references(mj_model=self.frozen_mj_model)
        segment_capsule_geoms = [
            self.frozen_mj_model.geom(geom_id)
            for geom_id in self._get_segment_capsule_geom_ids(
                mj_model=self.frozen_mj_model
            )
        ]
        self._segment_capsule_areas = np.array(
            [
                (np.pi * geom.size[0] ** 2) + (2 * geom.size[0] * 2 * geom.size[1])
                for geom in segment_capsule_geoms
            ]
        )
        self._disk_area = (
            np.pi
            * self.frozen_mj_model.geom(
                "BrittleStarMorphology/central_disk_pentagon_collider"
            ).size[0]
            ** 2
        )

    @property
    def environment_configuration(
        self,
    ) -> BrittleStarLightEscapeEnvironmentConfiguration:
        return super(MJCEnv, self).environment_configuration

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: MJCFBrittleStarMorphology,
        arena: MJCFAquariumArena,
        configuration: BrittleStarLightEscapeEnvironmentConfiguration,
    ) -> BrittleStarLightEscapeMJCEnvironment:
        assert arena.arena_configuration.sand_ground_color, (
            "This environment requires the 'sand_ground_color' "
            "parameter of the AquariumArenaConfiguration to be "
            "set to 'True'."
        )
        return super().from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=configuration
        )

    @staticmethod
    def _get_x_distance_from_start_position(state: MJCEnvState) -> float:
        disk_body_id = state.mj_model.body("BrittleStarMorphology/central_disk").id
        x_disk_position = state.mj_data.body(disk_body_id).xpos[0]
        start_x_position = BrittleStarLightEscapeEnvironmentBase._get_x_start_position(
            mj_model=state.mj_model
        )
        return x_disk_position - start_x_position

    @staticmethod
    def _get_light_value_at_xy_positions(
        state: MJCEnvState, xy_positions: np.array
    ) -> np.array:
        arena_size = np.array(state.mj_model.geom("groundplane").size[:2])

        shifted_xy_positions = xy_positions + arena_size
        normalized_xy_positions = shifted_xy_positions / (2 * arena_size)

        # Positive Y axis in light map and in world are inverted
        normalized_xy_positions[:, 1] = 1 - normalized_xy_positions[:, 1]
        yx_light_map_positions = normalized_xy_positions[:, ::-1] * np.array(
            [state.info["_light_map"].shape]
        )

        # Just take the closest light map value (should be enough precision)
        light_map_coords = np.round(yx_light_map_positions).T.astype(int)
        y_coords = np.clip(
            a=light_map_coords[0], a_min=0, a_max=state.info["_light_map"].shape[0] - 1
        )
        x_coords = np.clip(
            a=light_map_coords[1], a_min=0, a_max=state.info["_light_map"].shape[1] - 1
        )

        return state.info["_light_map"][y_coords, x_coords]

    def _get_light_per_segment(self, state: MJCEnvState) -> np.ndarray:
        segment_xy_positions = state.mj_data.geom_xpos[
            self._get_segment_capsule_geom_ids(mj_model=state.mj_model), :2
        ]

        values = self._get_light_value_at_xy_positions(
            state=state, xy_positions=segment_xy_positions
        )

        return values

    def _get_disk_light_income(self, state: MJCEnvState) -> float:
        # get disk position
        disk_xy_position = state.mj_data.xpos[
            state.mj_model.body("BrittleStarMorphology/central_disk").id
        ][:2]
        values = self._get_light_value_at_xy_positions(
            state=state, xy_positions=disk_xy_position[None, :]
        )
        return values[0]

    def _create_observables(self) -> List[MJCObservable]:
        base_observables = get_base_brittle_star_observables(
            mj_model=self.frozen_mj_model, backend="mjc"
        )

        num_segments = len(
            self._get_segment_capsule_geom_ids(mj_model=self.frozen_mj_model)
        )
        segment_light_intake_observable = MJCObservable(
            name="segment_light_intake",
            low=np.zeros(num_segments),
            high=np.ones(num_segments),
            retriever=lambda state: self._get_light_per_segment(state),
        )
        return base_observables + [segment_light_intake_observable]

    @staticmethod
    def _get_time(state: MJCEnvState) -> float:
        return state.mj_data.time

    def _get_mj_models_and_datas_to_render(
        self, state: MJCEnvState
    ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        mj_models, mj_datas = super()._get_mj_models_and_datas_to_render(state=state)
        self._update_mj_models_tex_data(mj_models=mj_models, state=state)
        if self.environment_configuration.color_contacts:
            self._color_segment_capsule_contacts(
                mj_models=mj_models, contact_bools=state.observations["segment_contact"]
            )
        return mj_models, mj_datas

    def get_renderer_context(
        self, renderer: Union[MujocoRenderer, mujoco.Renderer]
    ) -> mujoco.MjrContext:
        return super(MJCEnv, self).get_renderer_context(renderer)

    def get_renderer(
        self,
        identifier: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        state: MJCEnvState,
    ) -> Union[MujocoRenderer, mujoco.Renderer]:
        renderer = super().get_renderer(
            identifier=identifier, model=model, data=data, state=state
        )
        self._update_renderer_context(mj_model=model, state=state, renderer=renderer)
        return renderer

    def render(self, state: MJCEnvState) -> list[RenderFrame] | None:
        render_output = super().render(state=state)
        state.info["_light_map_has_changed"] = False
        return render_output

    def _update_light_map(self, state: MJCEnvState) -> MJCEnvState:
        ground_texture = state.mj_model.texture("groundplane")
        h, w = ground_texture.height[0], ground_texture.width[0]

        # Linear transition from bright to dark
        light_map = np.ones((h, w))
        light_map[:] = np.linspace(1, 0, w)

        if self.environment_configuration.light_perlin_noise_scale > 0:
            # Adds noise to the light map, while attempting to maintain a general decrease of brightness over the x-axis
            # We won't add noise to the starting zone
            starting_zone_width = w // 5

            # Generate perlin noise
            seed = state.rng.randint(low=0, high=10000)
            res = (self.environment_configuration.light_perlin_noise_scale,) * 2
            light_noise = generate_perlin_noise_2d(
                shape=light_map.shape, res=res, rng_key=seed, npi=np
            )
            light_noise = 1 - 2 * light_noise  # Rescale to [-1, 1] range

            # increasing noise weights over width (x-axis)
            light_noise_weights = np.zeros_like(light_noise)
            light_noise_weights[:, starting_zone_width:] = np.linspace(
                0, 1, w - starting_zone_width
            )

            light_noise = light_noise * light_noise_weights

            # Light noise interval from [-noise, noise] to [1 - noise, 1 + noise]
            light_noise = 1 + light_noise

            light_map *= light_noise

            # normalize light map
            light_map = (light_map - np.min(light_map)) / (
                np.max(light_map) - np.min(light_map)
            )

        info = state.info
        info.update({"_light_map": light_map, "_light_map_has_changed": True})
        # noinspection PyUnresolvedReferences
        return state.replace(info=info)

    def _finish_reset(
        self,
        models_and_datas: Tuple[mujoco.MjModel, mujoco.MjData],
        rng: np.random.RandomState,
    ) -> MJCEnvState:
        mj_model, mj_data = models_and_datas
        mujoco.mj_forward(m=mj_model, d=mj_data)

        # noinspection PyArgumentList
        state = MJCEnvState(
            mj_model=mj_model,
            mj_data=mj_data,
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

    def reset(self, rng: np.random.RandomState, *args, **kwargs) -> MJCEnvState:
        mj_model, mj_data = self._prepare_reset()

        # Set morphology position
        morphology_qpos_adr = mj_model.joint(
            "BrittleStarMorphology/freejoint/"
        ).qposadr[0]
        morphology_pos = np.array(
            [
                self._get_x_start_position(mj_model=mj_model),
                0.0,
                0.11,
            ]
        )

        mj_data.qpos[morphology_qpos_adr : morphology_qpos_adr + 3] = morphology_pos

        if self.environment_configuration.random_initial_rotation:
            z_axis_rotation = rng.uniform(-np.pi, np.pi)
            quat = euler2quat(0, 0, z_axis_rotation, axes="sxyz")
            mj_data.qpos[morphology_qpos_adr + 3 : morphology_qpos_adr + 7] = quat

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

        state = self._update_light_map(state=state)

        return state
