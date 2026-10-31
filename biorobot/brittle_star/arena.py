"""The aquarium arena: a walled, water-filled tank with an optional target and a sand-coloured (light map) floor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import mujoco
import numpy as np

from biorobot.core import Arena, euler_to_quat
from biorobot.utils import colors

WALL_RGBA = np.asarray([115, 147, 179, 50]) / 255
TARGET_RADIUS = 0.2


@dataclass
class AquariumArenaConfiguration:
    """Configuration of the aquarium.

    Args:
        name: Model name of the arena.
        size: Half-extents ``(x, y)`` of the floor in metres.
        sand_ground_color: Use a flat, sand-coloured floor texture. Required by the light escape task, which paints its
            light map onto this texture.
        light_map_resolution: Texels per half-extent (i.e. per 2 m) of the sand floor texture and thus of the light
            map: the texture is ``resolution * size`` texels across the ``2 * size`` m floor, so the default of 50
            gives 25 texels per metre and a 150 x 250 light map.
        attach_target: Add a movable (mocap) target, required by the directed locomotion task.
        wall_height: Half-height of the walls in metres.
        wall_thickness: Half-thickness of the walls in metres.
    """

    name: str = "AquariumArena"
    size: Tuple[int, int] = (5, 3)
    sand_ground_color: bool = False
    light_map_resolution: int = 50
    attach_target: bool = False
    wall_height: float = 1.5
    wall_thickness: float = 0.1


class AquariumArena(Arena):
    """Water (density 1000 kg/m^3, viscosity 0.0009 Pa s) in a walled tank.

    Collisions only happen between the morphology (contype 1) and the floor and invisible wall planes (conaffinity 1).
    """

    def __init__(self, configuration: AquariumArenaConfiguration | None = None) -> None:
        self.configuration = configuration or AquariumArenaConfiguration()
        super().__init__(name=self.configuration.name)

    def _build(self) -> None:
        self.spec.worldbody.add_camera(
            name="top_camera", pos=[0, 0, 20], quat=[1, 0, 0, 0]
        )
        self._configure_lights()
        # Water: enables MuJoCo's inertia-based fluid model (drag and buoyancy).
        self.spec.option.density = 1000
        self.spec.option.viscosity = 0.0009
        self.spec.add_texture(
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
            rgb1=[1.0, 1.0, 1.0],
            rgb2=[1.0, 1.0, 1.0],
            width=200,
            height=200,
        )
        self._build_ground()
        self._build_walls()
        if self.configuration.attach_target:
            self._build_target()

    def _configure_lights(self) -> None:
        self.spec.worldbody.add_light(
            pos=[-20, 0, 20],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            dir=[0, 0, -0.5],
            diffuse=[0.1, 0.1, 0.1],
            castshadow=False,
        )
        headlight = self.spec.visual.headlight
        headlight.ambient = [0.4, 0.4, 0.4]
        headlight.diffuse = [0.8, 0.8, 0.8]
        headlight.specular = [0.1, 0.1, 0.1]

    def _build_ground(self) -> None:
        w, h = self.configuration.size
        if self.configuration.sand_ground_color:
            resolution = self.configuration.light_map_resolution
            self.spec.add_texture(
                name="groundplane",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                rgb1=colors.rgba_sand[:3],
                width=resolution * w,
                height=resolution * h,
            )
            material = self.spec.add_material(name="groundplane", reflectance=0.0)
            rgba = colors.rgba_sand
        else:
            self.spec.add_texture(
                name="groundplane",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                rgb1=[0.2, 0.3, 0.4],
                rgb2=[0.1, 0.2, 0.3],
                width=200,
                height=200,
                mark=mujoco.mjtMark.mjMARK_EDGE,
                markrgb=[0.8, 0.8, 0.8],
            )
            # texrepeat 2 with texuniform makes the squares exactly 1x1 m.
            material = self.spec.add_material(
                name="groundplane", texrepeat=[2, 2], texuniform=True, reflectance=0.2
            )
            rgba = [0.5, 0.5, 0.5, 1.0]
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"

        self.spec.worldbody.add_geom(
            name="groundplane",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            material="groundplane",
            rgba=rgba,
            size=[w, h, 0.25],
            condim=4,
            contype=0,
            conaffinity=1,
        )

    def _build_walls(self) -> None:
        """Visual (transparent) wall boxes plus invisible inward-facing planes that act as the actual colliders."""
        (w, h), t, z = (
            self.configuration.size,
            self.configuration.wall_thickness,
            self.configuration.wall_height,
        )
        walls = (
            ("north_wall", [w, t, z], [0.0, h + t, z], [np.pi / 2, 0, 0]),
            ("south_wall", [w, t, z], [0.0, -h - t, z], [-np.pi / 2, 0, 0]),
            ("east_wall", [t, h + 2 * t, z], [-w - t, 0.0, z], [0, np.pi / 2, 0]),
            ("west_wall", [t, h + 2 * t, z], [w + t, 0.0, z], [0, -np.pi / 2, 0]),
        )
        for name, size, pos, _ in walls:
            self.spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size,
                pos=pos,
                rgba=WALL_RGBA,
                contype=0,
                conaffinity=0,
            )
        for name, size, pos, euler in walls:
            self.spec.worldbody.add_geom(
                name=f"{name}_collider",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=size,
                pos=pos,
                quat=euler_to_quat(np.array(euler)),
                rgba=[0, 0, 0, 0],
                contype=0,
                conaffinity=1,
            )

    def _build_target(self) -> None:
        # A mocap body, so every (vmapped) episode can place it via data.mocap_pos without changing the model.
        target = self.spec.worldbody.add_body(name="target", mocap=True)
        target.add_site(
            name="target_site",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[TARGET_RADIUS, 0, 0],
            rgba=[1.0, 0.0, 0.0, 1.0],
        )
