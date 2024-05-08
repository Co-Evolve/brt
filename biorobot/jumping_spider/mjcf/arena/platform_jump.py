from typing import Tuple

import numpy as np
from dm_control.mjcf.element import _ElementImpl
from moojoco.mjcf.arena import ArenaConfiguration, MJCFArena

from biorobot.utils import colors


class PlatformJumpArenaConfiguration(ArenaConfiguration):
    def __init__(
            self,
            name: str = "PlatformJumpArena",
            platform_size: Tuple[float, float] = (3, 3),
            gap: float = 5
    ) -> None:
        super().__init__(name=name)
        self.platform_size = platform_size
        self.gap = gap


class MJCFPlatformJumpArena(MJCFArena):
    @property
    def arena_configuration(self) -> PlatformJumpArenaConfiguration:
        return super().arena_configuration

    def _build(self, *args, **kwargs) -> None:
        self._configure_lights()
        self._configure_sky()
        self._build_platforms()
        self._configure_cameras()

    def _configure_cameras(self) -> None:
        middle = np.mean([self._start_platform.pos[0], self._end_platform.pos[0]])
        self.mjcf_model.worldbody.add(
            "camera", name="top_camera", pos=[middle, 0, 20], quat=[1, 0, 0, 0]
        )
        self.mjcf_model.worldbody.add(
            "camera", name="side_camera", pos=[middle, -15, 15], xyaxes=(1, 0, 0, 0, 0.5, 0.5)
        )

    def _configure_lights(self) -> None:
        self.mjcf_model.worldbody.add(
            "light",
            pos=[-20, 0, 20],
            directional=True,
            dir=[0, 0, -0.5],
            diffuse=[0.1, 0.1, 0.1],
            castshadow=False,
        )
        self.mjcf_model.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8], specular=[0.1, 0.1, 0.1]
        )

    def _configure_sky(self) -> None:
        # white sky
        self.mjcf_model.asset.add(
            "texture",
            type="skybox",
            builtin="flat",
            rgb1="1.0 1.0 1.0",
            rgb2="1.0 1.0 1.0",
            width=200,
            height=200,
        )

    def _build_platform(self, x_pos: float, name: str) -> _ElementImpl:
        return self.mjcf_body.add(
            "geom",
            type="box",
            name=f"{self.base_name}_{name}",
            rgba=colors.rgba_gray,
            pos=[x_pos, 0, -0.25],
            size=list(self.arena_configuration.platform_size) + [0.25],
            contype=0,
            conaffinity=1,
        )

    def _build_platforms(self) -> None:
        self._start_platform = self._build_platform(
            x_pos=0.0,
            name="start_platform"
        )
        self._end_platform = self._build_platform(
            x_pos=self.arena_configuration.platform_size[0] + self.arena_configuration.gap,
            name="end_platform"
        )
