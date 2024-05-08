from typing import Sequence

import numpy as np
from dm_control.mjcf.element import _ElementImpl

from biorobot.jumping_spider.mjcf.arena.shared import SpiderArenaConfiguration, MJCFSpiderArena
from biorobot.utils import colors


class PlatformJumpArenaConfiguration(SpiderArenaConfiguration):
    def __init__(
            self,
            name: str = "PlatformJumpArena",
            start_platform_pos: Sequence[float] = (0, 0, -0.125),
            start_platform_size: Sequence[float] = (4, 3, 0.25),
            offset_to_end_platform: Sequence[float] = (10, 0, 0),
            end_platform_size: Sequence[float] = (3, 3, 0.25)
    ) -> None:
        super().__init__(name=name)
        self.start_platform_pos = start_platform_pos
        self.start_platform_size = start_platform_size
        self.offset_to_end_platform = offset_to_end_platform
        self.end_platform_size = end_platform_size


class MJCFPlatformJumpArena(MJCFSpiderArena):
    @property
    def arena_configuration(self) -> PlatformJumpArenaConfiguration:
        return super().arena_configuration

    def _build(self, *args, **kwargs) -> None:
        self._build_platforms()
        super()._build(args=args, kwargs=kwargs)

    def _configure_cameras(self) -> None:
        middle = np.mean([self._start_platform.pos[0], self._end_platform.pos[0]])
        self.mjcf_model.worldbody.add(
            "camera", name="top_camera", pos=[middle, 0, 20], quat=[1, 0, 0, 0]
        )
        self.mjcf_model.worldbody.add(
            "camera", name="side_camera", pos=[middle, -15, 15], xyaxes=(1, 0, 0, 0, 0.5, 0.5)
        )

    def _build_platform(self, pos: Sequence[float], size: Sequence[float],
                        name: str) -> _ElementImpl:
        return self.mjcf_body.add(
            "geom",
            type="box",
            name=f"{self.base_name}_{name}",
            rgba=colors.rgba_gray,
            pos=pos,
            size=np.array(size) / 2,
            contype=0,
            conaffinity=1,
        )

    def _build_platforms(self) -> None:
        self._start_platform = self._build_platform(
            pos=self.arena_configuration.start_platform_pos,
            size=self.arena_configuration.start_platform_size,
            name="start_platform"
        )
        self._end_platform = self._build_platform(
            pos=np.array(self.arena_configuration.start_platform_pos) + np.array(
                self.arena_configuration.offset_to_end_platform),
            size=self.arena_configuration.end_platform_size,
            name="end_platform"
        )
