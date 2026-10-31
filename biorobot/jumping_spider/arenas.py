"""Arenas of the jumping spider tasks.

The spider always starts at the origin, facing +x, standing on a surface whose top is at ``z = 0``. Every arena has a
``dragline_anchor`` site behind the start position (used when the morphology has a dragline), a ``top_camera`` and a
``side_camera``. Task objects whose pose changes per episode (the end platform, the target) are mocap bodies, so that
each (vmapped) episode can place them independently through ``data.mocap_pos``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import mujoco
import numpy as np

from biorobot.core.builder import Arena
from biorobot.jumping_spider import colors
from biorobot.jumping_spider.morphology import xyaxes_to_quat


@dataclass
class JumpingSpiderArenaConfiguration:
    """Settings shared by all jumping spider arenas.

    Args:
        name: Arena name (also the MjSpec model name).
        dragline_anchor_position: Position (m) of the dragline's anchor; behind the start position, on the ground.
    """

    name: str = "JumpingSpiderArena"
    dragline_anchor_position: Tuple[float, float, float] = (-0.15, 0.0, 0.0)


class JumpingSpiderArena(Arena):
    def __init__(self, configuration: JumpingSpiderArenaConfiguration) -> None:
        self.configuration = configuration
        super().__init__(name=configuration.name)

    def _build(self) -> None:
        self.spec.default.geom.contype = 0
        self.spec.default.geom.conaffinity = 1
        self.spec.default.geom.condim = 3
        self._configure_visuals()
        self.spec.worldbody.add_site(
            name="dragline_anchor",
            pos=self.configuration.dragline_anchor_position,
            size=[0.003, 0, 0],
            rgba=colors.SITE,
        )
        self._build_task_objects()
        center = self._camera_center()
        self.spec.worldbody.add_camera(
            name="top_camera", pos=[center, 0, 1.2], quat=[1, 0, 0, 0]
        )
        self.spec.worldbody.add_camera(
            name="side_camera",
            pos=[center, -1.2, 0.4],
            quat=xyaxes_to_quat([1, 0, 0], [0, 0.3, 1]),
        )

    def _build_task_objects(self) -> None:
        """Build the ground and task-specific objects."""
        raise NotImplementedError

    def _camera_center(self) -> float:
        """x-coordinate that the world cameras look at."""
        return 0.0

    def _configure_visuals(self) -> None:
        self.spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
        self.spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
        self.spec.visual.headlight.specular = [0.1, 0.1, 0.1]
        self.spec.worldbody.add_light(
            pos=[0, 0, 2],
            dir=[0, 0, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.3, 0.3, 0.3],
            castshadow=False,
        )
        self.spec.add_texture(
            name="sky",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
            rgb1=[1, 1, 1],
            rgb2=[1, 1, 1],
            width=64,
            height=64,
        )

    def _add_grid_material(
        self, name: str, rgb: np.ndarray, tile_size: float = 0.1
    ) -> str:
        """A checker material whose tiles are ``tile_size`` metres wide (handy to read off jump distances)."""
        self.spec.add_texture(
            name=name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=rgb,
            rgb2=0.85 * rgb,
            width=64,
            height=64,
            mark=mujoco.mjtMark.mjMARK_EDGE,
            markrgb=[1, 1, 1],
        )
        material = self.spec.add_material(name=name, texuniform=True, reflectance=0.1)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
        # With texuniform, MuJoCo repeats the texture texrepeat times per 2 m; one checker texture has 2x2 tiles.
        material.texrepeat = [1 / tile_size, 1 / tile_size]
        return name

    def _add_floor(self, z: float = 0.0, half_size: float = 5.0) -> None:
        material = self._add_grid_material("floor", np.array([0.3, 0.35, 0.4]))
        self.spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0, 0, z],
            size=[half_size, half_size, 0.05],
            material=material,
        )


# ----------------------------------------------------------------------------------------------------------------------
# Long jump
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class LongJumpArenaConfiguration(JumpingSpiderArenaConfiguration):
    """A straight track on flat ground.

    Args:
        track_length: Length of the track (m), starting just behind the spider.
        track_width: Width of the track (m). Leaving it sideways ends the episode.
    """

    name: str = "LongJumpArena"
    track_length: float = 2.0
    track_width: float = 0.4


class LongJumpArena(JumpingSpiderArena):
    configuration: LongJumpArenaConfiguration

    def _build_task_objects(self) -> None:
        config = self.configuration
        start = -0.2
        material = self._add_grid_material("track", colors.TRACK[:3])
        # Planes collide over their full (infinite) extent; the size only sets what is drawn.
        self.spec.worldbody.add_geom(
            name="track",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[start + config.track_length / 2, 0, 0],
            size=[config.track_length / 2, config.track_width / 2, 0.05],
            material=material,
        )

    def _camera_center(self) -> float:
        return 0.4


# ----------------------------------------------------------------------------------------------------------------------
# Platform jump
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class PlatformJumpArenaConfiguration(JumpingSpiderArenaConfiguration):
    """A start platform and an end platform separated by a gap; the floor lies ``platform_height`` below their tops.

    Args:
        start_platform_size: Full (x, y) extent of the start platform (m). Its front edge is at ``start_platform_edge``.
        start_platform_edge: x-coordinate of the start platform's front edge (m).
        end_platform_size: Full (x, y) extent of the end platform (m).
        platform_height: Height of both platforms above the floor (m).
        default_gap: Gap between the platforms in the compiled model (m); environments randomise it per episode.
    """

    name: str = "PlatformJumpArena"
    start_platform_size: Tuple[float, float] = (0.4, 0.4)
    start_platform_edge: float = 0.1
    end_platform_size: Tuple[float, float] = (0.3, 0.4)
    platform_height: float = 0.1
    default_gap: float = 0.15


class PlatformJumpArena(JumpingSpiderArena):
    configuration: PlatformJumpArenaConfiguration

    def _build_task_objects(self) -> None:
        config = self.configuration
        height = config.platform_height
        self._add_floor(z=-height)

        start_length, start_width = config.start_platform_size
        self.spec.worldbody.add_geom(
            name="start_platform",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[config.start_platform_edge - start_length / 2, 0, -height / 2],
            size=[start_length / 2, start_width / 2, height / 2],
            rgba=colors.PLATFORM,
        )

        # The end platform's body frame is the centre of its top surface.
        end_length, end_width = config.end_platform_size
        end_platform = self.spec.worldbody.add_body(
            name="end_platform",
            mocap=True,
            pos=[self.end_platform_x(config.default_gap), 0, 0],
        )
        end_platform.add_geom(
            name="end_platform",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, 0, -height / 2],
            size=[end_length / 2, end_width / 2, height / 2],
            rgba=colors.END_PLATFORM,
        )

    def end_platform_x(self, gap: float):
        """x-coordinate of the end platform's centre for a given ``gap`` (works on numpy and jax values)."""
        config = self.configuration
        return config.start_platform_edge + gap + config.end_platform_size[0] / 2

    def _camera_center(self) -> float:
        return self.end_platform_x(self.configuration.default_gap) / 2


# ----------------------------------------------------------------------------------------------------------------------
# Directed jump
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class DirectedJumpArenaConfiguration(JumpingSpiderArenaConfiguration):
    """Flat ground with a (non-colliding) spherical target.

    Args:
        target_radius: Radius of the target sphere (m); reaching it with the cephalothorax centre ends the episode.
    """

    name: str = "DirectedJumpArena"
    target_radius: float = 0.03


class DirectedJumpArena(JumpingSpiderArena):
    configuration: DirectedJumpArenaConfiguration

    def _build_task_objects(self) -> None:
        self._add_floor()
        target = self.spec.worldbody.add_body(
            name="target", mocap=True, pos=[0.3, 0, 0.1]
        )
        target.add_geom(
            name="target",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.configuration.target_radius, 0, 0],
            rgba=colors.TARGET,
            contype=0,
            conaffinity=0,
        )

    def _camera_center(self) -> float:
        return 0.2
