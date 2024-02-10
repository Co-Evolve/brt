from typing import Tuple

from moojoco.mjcf.arena import ArenaConfiguration, MJCFArena
from transforms3d.euler import euler2quat

from biorobot.toy_example.mjcf.arena.target import MJCFTarget


class PlaneWithTargetArenaConfiguration(ArenaConfiguration):
    def __init__(self, size: Tuple[int, int] = (8, 8)) -> None:
        super().__init__(name="PlaneWithTargetArena")
        self.size = size


class MJCFPlaneWithTargetArena(MJCFArena):
    @property
    def arena_configuration(self) -> PlaneWithTargetArenaConfiguration:
        return super().arena_configuration

    def _build(
        self,
    ) -> None:
        self._build_floor()
        self._configure_lights()
        self._configure_top_camera()
        self.target = self._attach_target()

    def _build_floor(self) -> None:
        self._ground_texture = self.mjcf_model.asset.add(
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
        self._ground_material = self.mjcf_model.asset.add(
            "material",
            name="groundplane",
            texrepeat=[2, 2],  # Makes white squares exactly 1x1 length units.
            texuniform=True,
            reflectance=0.2,
            texture=self._ground_texture,
        )
        # Build groundplane.
        self._ground_geom = self.mjcf_body.add(
            "geom",
            type="plane",
            name="groundplane",
            material=self._ground_material,
            size=list(self.arena_configuration.size) + [0.25],
            condim=3,
            friction=[1, 0.5, 0.5],
            contype=0,
            conaffinity=1,
        )

    def _configure_lights(self) -> None:
        self.mjcf_model.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8], specular=[0.1, 0.1, 0.1]
        )

    def _configure_top_camera(self) -> None:
        self._main_camera = self.mjcf_body.add(
            "camera",
            name="main_camera",
            pos=[0, 0, 10],
            quat=euler2quat(0, 0, 0),
        )

    def _attach_target(self) -> MJCFTarget:
        target = MJCFTarget(parent=self, name="target")
        return target
