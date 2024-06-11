from typing import Sequence

import numpy as np
from dm_control.mjcf.element import _ElementImpl

from biorobot.brittle_star.mjcf.arena.target import MJCFTarget
from biorobot.jumping_spider.mjcf.arena.shared import SpiderArenaConfiguration, MJCFSpiderArena


class DirectedJumpArenaConfiguration(SpiderArenaConfiguration):
    def __init__(
            self,
            name: str = "PlatformJumpArena",
            platform_size: Sequence[float] = (4, 3, 0.25),
    ) -> None:
        super().__init__(name=name)
        self.platform_size = platform_size


class MJCFDirectedJumpArena(MJCFSpiderArena):
    @property
    def arena_configuration(self) -> DirectedJumpArenaConfiguration:
        return super().arena_configuration

    def _build(self, *args, **kwargs) -> None:
        self._build_platform()
        self._build_target()
        super()._build(args=args, kwargs=kwargs)

    def _configure_cameras(self) -> None:
        self.mjcf_model.worldbody.add(
            "camera", name="top_camera", pos=[0, 0, 20], quat=[1, 0, 0, 0]
        )
        self.mjcf_model.worldbody.add(
            "camera", name="side_camera", pos=[2.5, -15, 2.5], xyaxes=(1, 0, 0, 0, 0, 1)
        )

    def _build_platform(self) -> _ElementImpl:
        platform_texture = self.mjcf_model.asset.add(
            "texture",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            type="2d",
            builtin="checker",
            name="groundplane",
            width=200,
            height=200,
            mark="edge",
            markrgb=[0.8, 0.8, 0.8],
        )
        platform_material = self.mjcf_model.asset.add(
            "material",
            name="groundplane",
            texrepeat=[2, 2],  # Makes white squares exactly 1x1 length units.
            texuniform=True,
            reflectance=0.2,
            texture=platform_texture,
        )
        self._platform = self.mjcf_body.add(
            "geom",
            type="box",
            name=f"{self.base_name}_platform",
            material=platform_material,
            pos=np.array([0, 0, -self.arena_configuration.platform_size[2] / 2]),
            size=self.arena_configuration.platform_size,
            contype=0,
            conaffinity=1,
        )

    def _build_target(self) -> None:
        self._target = MJCFTarget(parent=self, name="target")
