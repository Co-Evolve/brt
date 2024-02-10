from typing import Tuple

import numpy as np
from moojoco.mjcf.arena import ArenaConfiguration, MJCFArena

from biorobot.toy_example.mjcf.arena.target import MJCFTarget
from biorobot.utils import colors


class AquariumArenaConfiguration(ArenaConfiguration):
    def __init__(
        self,
        name: str = "AquariumArena",
        size: Tuple[int, int] = (10, 5),
        sand_ground_color: bool = False,
        attach_target: bool = False,
        wall_height: float = 1.5,
        wall_thickness: float = 0.1,
    ) -> None:
        super().__init__(name=name)
        self.size = size
        self.sand_ground_color = sand_ground_color
        self.attach_target = attach_target
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness


class MJCFAquariumArena(MJCFArena):
    @property
    def arena_configuration(self) -> AquariumArenaConfiguration:
        return super().arena_configuration

    def _build(self, *args, **kwargs) -> None:
        self._configure_cameras()
        self._configure_lights()
        self._configure_water()
        self._configure_sky()
        self._build_ground()
        self._build_walls()
        self._build_target()

    def _configure_cameras(self) -> None:
        self.mjcf_model.worldbody.add(
            "camera", name="top_camera", pos=[0, 0, 20], quat=[1, 0, 0, 0]
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

    def _configure_water(self) -> None:
        self.mjcf_model.option.density = 1000
        self.mjcf_model.option.viscosity = 0.0009

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

    def _build_ground(self) -> None:
        if self.arena_configuration.sand_ground_color:
            ground_texture = self.mjcf_model.asset.add(
                "texture",
                type="2d",
                builtin="flat",
                name="groundplane",
                rgb1=colors.rgba_sand[:3],
                width=200,
                height=200,
            )
            ground_material = self.mjcf_model.asset.add(
                "material",
                name="groundplane",
                reflectance=0.0,
                texture=ground_texture,
            )
            rgba = colors.rgba_sand
        else:
            ground_texture = self.mjcf_model.asset.add(
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
            ground_material = self.mjcf_model.asset.add(
                "material",
                name="groundplane",
                texrepeat=[2, 2],  # Makes white squares exactly 1x1 length units.
                texuniform=True,
                reflectance=0.2,
                texture=ground_texture,
            )
            rgba = None
        # Build groundplane.
        self._ground_geom = self.mjcf_body.add(
            "geom",
            type="plane",
            name="groundplane",
            material=ground_material,
            rgba=rgba,
            size=list(self.arena_configuration.size) + [0.25],
            condim=3,
            friction=[1, 0.5, 0.5],
            contype=0,
            conaffinity=1,
        )

    def _build_walls(self) -> None:
        wall_rgba = np.asarray([115, 147, 179, 50]) / 255

        self.mjcf_model.worldbody.add(
            "geom",
            type="box",
            name="north_wall",
            size=[
                self.arena_configuration.size[0],
                self.arena_configuration.wall_thickness,
                self.arena_configuration.wall_height,
            ],
            pos=[
                0.0,
                self.arena_configuration.size[1]
                + self.arena_configuration.wall_thickness,
                self.arena_configuration.wall_height,
            ],
            rgba=wall_rgba,
            contype=0,
            conaffinity=1,
        )
        self.mjcf_model.worldbody.add(
            "geom",
            type="box",
            name="south_wall",
            size=[
                self.arena_configuration.size[0],
                self.arena_configuration.wall_thickness,
                self.arena_configuration.wall_height,
            ],
            pos=[
                0.0,
                -self.arena_configuration.size[1]
                - self.arena_configuration.wall_thickness,
                self.arena_configuration.wall_height,
            ],
            rgba=wall_rgba,
            contype=0,
            conaffinity=1,
        )
        self.mjcf_model.worldbody.add(
            "geom",
            type="box",
            name="east_wall",
            size=[
                self.arena_configuration.wall_thickness,
                self.arena_configuration.size[1]
                + 2 * self.arena_configuration.wall_thickness,
                self.arena_configuration.wall_height,
            ],
            pos=[
                -self.arena_configuration.size[0]
                - self.arena_configuration.wall_thickness,
                0.0,
                self.arena_configuration.wall_height,
            ],
            rgba=wall_rgba,
            contype=0,
            conaffinity=1,
        )
        self.mjcf_model.worldbody.add(
            "geom",
            type="box",
            name="west_wall",
            size=[
                self.arena_configuration.wall_thickness,
                self.arena_configuration.size[1]
                + 2 * self.arena_configuration.wall_thickness,
                self.arena_configuration.wall_height,
            ],
            pos=[
                self.arena_configuration.size[0]
                + self.arena_configuration.wall_thickness,
                0.0,
                self.arena_configuration.wall_height,
            ],
            rgba=wall_rgba,
            contype=0,
            conaffinity=1,
        )

    def _build_target(self) -> None:
        if self.arena_configuration.attach_target:
            self._target = MJCFTarget(parent=self, name="target")


if __name__ == "__main__":
    MJCFAquariumArena(
        AquariumArenaConfiguration(sand_ground_color=True)
    ).export_to_xml_with_assets("./mjcf")
