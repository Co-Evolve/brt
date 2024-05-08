from typing import Tuple

from moojoco.mjcf.arena import ArenaConfiguration, MJCFArena

from biorobot.utils import colors


class LongJumpArenaConfiguration(ArenaConfiguration):
    def __init__(
            self,
            name: str = "PlatformJumpArena",
            track_size: Tuple[float] = (10, 3)
    ) -> None:
        super().__init__(name=name)
        self.track_size = track_size


class MJCFLongJumpArena(MJCFArena):
    @property
    def arena_configuration(self) -> LongJumpArenaConfiguration:
        return super().arena_configuration

    def _build(self, *args, **kwargs) -> None:
        self._configure_lights()
        self._configure_sky()
        self._build_track()
        self._configure_cameras()

    def _configure_cameras(self) -> None:
        self.mjcf_model.worldbody.add(
            "camera", name="top_camera", pos=[self._track.pos[0], 0, 20], quat=[1, 0, 0, 0]
        )
        self.mjcf_model.worldbody.add(
            "camera", name="side_camera", pos=[self._track.pos[0], -15, 15], xyaxes=(1, 0, 0, 0, 0.5, 0.5)
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
