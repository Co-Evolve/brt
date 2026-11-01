"""The seahorse tail's arena: an empty white world with the tail's fixed attachment point, lights and cameras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import mujoco
import numpy as np

from biorobot.core.builder import Arena, euler_to_quat


@dataclass
class SeahorseArenaConfiguration:
    """Settings of the seahorse arena.

    Args:
        name: Arena name (also the MjSpec model name).
        attachment_position: Position (m) of the tail's base (segment 0), which is fixed to the world.
        attachment_euler: Orientation (xyz Euler angles, rad) of the tail's base; the default rotation by pi around x
            makes the tail hang downwards (-z) with its ventral side towards +x.
    """

    name: str = "SeahorseArena"
    attachment_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    attachment_euler: Tuple[float, float, float] = (np.pi, 0.0, 0.0)


class SeahorseArena(Arena):
    """Cameras (in id order): ``side_camera`` (looks at the sinistral side), ``corner``, ``ventral``, ``top-down`` and
    ``ventral-closeup``."""

    def __init__(self, configuration: SeahorseArenaConfiguration) -> None:
        self.configuration = configuration
        super().__init__(name=configuration.name)

    def _build(self) -> None:
        self.spec.worldbody.add_site(
            name="seahorse_attachment_site",
            pos=self.configuration.attachment_position,
            quat=euler_to_quat(np.asarray(self.configuration.attachment_euler)),
        )
        self._build_lights()
        self._build_cameras()
        self.spec.add_texture(
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
            rgb1=[1, 1, 1],
            rgb2=[1, 1, 1],
            width=200,
            height=200,
        )

    def _build_lights(self) -> None:
        self.spec.worldbody.add_light(
            name="corner",
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            pos=[0.43011626335213127, -0.43011626335213127, -0.175],
            dir=[1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            ambient=[0.1, 0.1, 0.1],
        )
        self.spec.worldbody.add_light(
            name="ventral",
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            pos=[0.6, 0, -0.175],
            dir=[-1, 0, 0],
            ambient=[0.01, 0.01, 0.01],
            specular=[0, 0, 0],
        )

    def _build_cameras(self) -> None:
        side = self.spec.worldbody.add_camera(
            name="side_camera", pos=[0.075, -0.6, -0.215]
        )
        side.alt.type = mujoco.mjtOrientation.mjORIENTATION_ZAXIS
        side.alt.zaxis = [0, -1, 0]
        for name, pos, xyaxes in (
            (
                "corner",
                [0.43011626335213127, -0.43011626335213127, -0.175],
                [0.5, 0.5, 0, 0, 0, 1],
            ),
            ("ventral", [0.55, 0.0, -0.175], [0, 1, 0, 0, 0, 1]),
            ("top-down", [0, 0, 0.5], [0, 1, 0, -1, 0, 0]),
            ("ventral-closeup", [0.25, 0, -0.02], [0, 1, 0, 0, 0, 1]),
        ):
            camera = self.spec.worldbody.add_camera(name=name, pos=pos)
            camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
            camera.alt.xyaxes = xyaxes
