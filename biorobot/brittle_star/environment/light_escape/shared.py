import abc
from typing import List, Union

import chex
import mujoco
import numpy as np
from moojoco.environment.base import BaseEnvState
from moojoco.environment.renderer import MujocoRenderer

from biorobot.brittle_star.environment.shared.base import (
    BrittleStarEnvironmentBase,
    BrittleStarEnvironmentBaseConfiguration,
)
from biorobot.utils import colors


class BrittleStarLightEscapeEnvironmentConfiguration(
    BrittleStarEnvironmentBaseConfiguration
):
    def __init__(
        self,
        light_perlin_noise_scale: int = 0,
        joint_randomization_noise_scale: float = 0.0,
        color_contacts: bool = False,
        *args,
        **kwargs,
    ) -> None:
        assert light_perlin_noise_scale == 0 or 200 % light_perlin_noise_scale == 0, (
            "Please only provide integer factors of 200 for the "
            "'light_perlin_noise_scale' parameter."
        )
        super().__init__(
            joint_randomization_noise_scale=joint_randomization_noise_scale,
            color_contacts=color_contacts,
            *args,
            **kwargs,
        )
        self.light_perlin_noise_scale = int(light_perlin_noise_scale)


class BrittleStarLightEscapeEnvironmentBase(BrittleStarEnvironmentBase):
    def __init__(self) -> None:
        super().__init__()

    def _update_reward(
        self, state: BaseEnvState, previous_state: BaseEnvState
    ) -> BaseEnvState:
        previous_average_light_income = self._get_average_light_income(
            state=previous_state
        )
        current_average_light_income = self._get_average_light_income(state=state)
        reward = previous_average_light_income - current_average_light_income

        # noinspection PyUnresolvedReferences
        return state.replace(reward=reward)

    def _update_info(self, state: BaseEnvState) -> BaseEnvState:
        info = state.info
        info.update(
            {
                "time": self._get_time(state=state),
                "x_distance_travelled": self._get_x_distance_from_start_position(
                    state=state
                ),
            }
        )
        # noinspection PyUnresolvedReferences
        return state.replace(info=info)

    def _update_terminated(self, state: BaseEnvState) -> BaseEnvState:
        # noinspection PyUnresolvedReferences
        return state.replace(terminated=False)

    def _update_truncated(self, state: BaseEnvState) -> BaseEnvState:
        truncated = (
            self._get_time(state=state) > self.environment_configuration.simulation_time
        )

        # noinspection PyUnresolvedReferences
        return state.replace(truncated=truncated)

    @staticmethod
    def _update_mj_models_tex_rgb(
        mj_models: List[mujoco.MjModel], state: BaseEnvState
    ) -> None:
        if np.any(state.info["_light_map_has_changed"]):
            ground_texture = state.mj_model.texture("groundplane")
            h, w, adr = (
                ground_texture.height[0],
                ground_texture.width[0],
                ground_texture.adr[0],
            )
            size = h * w * 3
            # Update textures in mj models
            light_maps = state.info["_light_map"]
            if len(mj_models) == 1:
                light_maps = [light_maps]

            for mj_model, light_map in zip(mj_models, light_maps):
                coloured_light_map = 0.3 + 0.7 * np.array(light_map)
                coloured_light_map = np.stack((coloured_light_map,) * 3, axis=-1)
                coloured_light_map = coloured_light_map * colors.rgba_sand[:3]
                mj_model.tex_rgb[adr : adr + size] = coloured_light_map.flatten() * 255

    def _update_renderer_context(
        self,
        mj_model: mujoco.MjModel,
        state: BaseEnvState,
        renderer: Union[MujocoRenderer, mujoco.Renderer],
    ) -> None:
        if np.any(state.info["_light_map_has_changed"]):
            ground_texture = mj_model.texture("groundplane")
            context = self.get_renderer_context(renderer=renderer)
            mujoco.mjr_uploadTexture(m=mj_model, con=context, texid=ground_texture.id)

    @staticmethod
    def _get_x_start_position(mj_model: mujoco.MjModel) -> float:
        arena_size = mj_model.geom("groundplane").size[0]
        return -0.75 * arena_size

    @property
    @abc.abstractmethod
    def environment_configuration(
        self,
    ) -> BrittleStarLightEscapeEnvironmentConfiguration:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _get_time(state: BaseEnvState) -> float:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _get_x_distance_from_start_position(state: BaseEnvState) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_light_per_segment(self, state: BaseEnvState) -> chex.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_average_light_income(self, state: BaseEnvState) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def get_renderer_context(
        self, renderer: Union[MujocoRenderer, mujoco.Renderer]
    ) -> mujoco.MjrContext:
        raise NotImplementedError
