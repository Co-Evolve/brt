"""The swimmer's arena: an unbounded, weightless fluid with a purely visual floor below the swimming plane."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from biorobot.core import Arena
from biorobot.swimmer import colors

INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}


@dataclass
class FluidArenaConfiguration:
    """Settings of the fluid arena.

    Args:
        name: Arena name (also the MjSpec model name).
        viscosity: Dynamic viscosity of the medium (Pa s). The default 0.1 Pa s is ~100 times that of water.
        density: Density of the medium (kg/m^3).
        floor_depth: Depth (m) of the visual floor below the swimming plane; it has no collisions.
        floor_size: Half-extent (m) of the visual floor.
        integrator: ``"euler"`` (MuJoCo's default) or ``"implicitfast"``. ``"implicitfast"`` is more robust for light
            or thin bodies and stiff servos, but MJX does not implement it with fluid forces, so it only works with the
            ``"mjc"`` and ``"warp"`` backends.
    """

    name: str = "FluidArena"
    viscosity: float = 0.1
    density: float = 1000.0
    floor_depth: float = 0.3
    floor_size: float = 10.0
    integrator: str = "euler"

    def __post_init__(self) -> None:
        if self.integrator not in INTEGRATORS:
            raise ValueError(
                f"integrator must be one of {sorted(INTEGRATORS)}, got {self.integrator!r}."
            )


class FluidArena(Arena):
    """The swimmer moves in the horizontal plane through a fluid, without gravity (the body is neutrally buoyant in
    effect). Fluid forces come from MuJoCo's fluid model; there is no flow field."""

    def __init__(self, configuration: FluidArenaConfiguration | None = None) -> None:
        self.configuration = configuration or FluidArenaConfiguration()
        super().__init__(name=self.configuration.name)

    def _build(self) -> None:
        option = self.spec.option
        option.gravity = [0, 0, 0]
        option.viscosity = self.configuration.viscosity
        option.density = self.configuration.density
        option.integrator = INTEGRATORS[self.configuration.integrator]

        visual = self.spec.visual
        visual.headlight.diffuse = [0.6, 0.6, 0.6]
        visual.headlight.ambient = [0.3, 0.3, 0.3]
        visual.headlight.specular = [0, 0, 0]
        visual.rgba.haze = colors.HAZE
        visual.global_.azimuth = 120
        visual.global_.elevation = -20

        self.spec.worldbody.add_light(
            name="top_light", pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8]
        )
        self.spec.add_texture(
            name="floor",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=colors.FLOOR_LIGHT,
            rgb2=colors.FLOOR_DARK,
            width=300,
            height=300,
            mark=mujoco.mjtMark.mjMARK_EDGE,
            markrgb=[0.8, 0.8, 0.8],
        )
        material = self.spec.add_material(
            name="floor", texrepeat=[4, 4], texuniform=True, reflectance=0.2
        )
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
        size = self.configuration.floor_size
        self.spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[size, size, 0.1],
            pos=[0, 0, -self.configuration.floor_depth],
            material="floor",
            contype=0,
            conaffinity=0,
        )
