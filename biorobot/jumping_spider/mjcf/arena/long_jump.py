from typing import Tuple

from biorobot.jumping_spider.mjcf.arena.shared import MJCFSpiderArena, SpiderArenaConfiguration
from biorobot.utils import colors


class LongJumpArenaConfiguration(SpiderArenaConfiguration):
    def __init__(
            self,
            name: str = "LongJumpArena",
            track_size: Tuple[float] = (10, 3)
    ) -> None:
        super().__init__(name=name)
        self.track_size = track_size


class MJCFLongJumpArena(MJCFSpiderArena):
    @property
    def arena_configuration(self) -> LongJumpArenaConfiguration:
        return super().arena_configuration

    def _build(self, *args, **kwargs) -> None:
        self._build_track()
        super()._build(args=args, kwargs=kwargs)

    def _configure_cameras(self) -> None:
        self.mjcf_model.worldbody.add(
            "camera", name="top_camera", pos=[self._track.pos[0], 0, 20], quat=[1, 0, 0, 0]
        )
        self.mjcf_model.worldbody.add(
            "camera", name="side_camera", pos=[self._track.pos[0], -15, 15], xyaxes=(1, 0, 0, 0, 0.5, 0.5)
        )

    def _build_track(self) -> None:
        ground_texture = self.mjcf_model.asset.add(
            "texture",
            rgb1=colors.rgba_red[:3],
            type="2d",
            builtin="flat",
            name=f"{self.base_name}_track_texture",
            width=200,
            height=200,
            mark="edge",
            markrgb=[1, 1, 1],
        )
        ground_material = self.mjcf_model.asset.add(
            "material",
            name=f"{self.base_name}_track_material",
            texrepeat=[self.arena_configuration.track_size[0] * 2, 1],  # Makes white squares exactly 1x1 length units.
            texuniform=False,
            reflectance=0.2,
            texture=ground_texture,
        )

        self._track = self.mjcf_body.add(
            "geom",
            type="plane",
            name=f"{self.base_name}_track_geom",
            material=ground_material,
            pos=[self.arena_configuration.track_size[0] - self.arena_configuration.track_size[1], 0, 0],
            size=list(self.arena_configuration.track_size) + [0.25],
            contype=0,
            conaffinity=1,
        )
