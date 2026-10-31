"""Light escape: move the whole body from the bright into the dark part of the aquarium.

The light map is a ``(height, width)`` array in ``[0, 1]`` over the floor (``light_map_resolution`` texels per 2 m):
it decreases linearly from bright (x = -size_x) to dark (x = +size_x), optionally perturbed with Perlin noise. It lives
in ``info["_light_map"]`` (so every vmapped episode has its own) and is painted onto the sand floor when rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np

from biorobot.brittle_star.arena import AquariumArena, AquariumArenaConfiguration
from biorobot.brittle_star.environment.base import (
    INITIAL_DISK_HEIGHT,
    BrittleStarEnv,
    BrittleStarEnvConfig,
    make_brittle_star_env,
)
from biorobot.brittle_star.specification import BrittleStarMorphologySpecification
from biorobot.core import EnvState, Observable
from biorobot.registry import register
from biorobot.utils import colors
from biorobot.utils.noise import generate_perlin_noise_2d


@dataclass
class LightEscapeEnvConfig(BrittleStarEnvConfig):
    """Args:
    light_perlin_noise_scale: Number of Perlin noise periods along each light map axis (0 disables noise). Must
        divide both light map dimensions (150 x 250 texels for the default arena).
    """

    light_perlin_noise_scale: int = 0


class LightEscapeEnv(BrittleStarEnv):
    """Reward: the decrease in body light income during the step.

    The body light income is the area-weighted mean of the light at the central disk (area of the disk's top) and at
    each segment capsule (projected area), each sampled with bilinear interpolation of the light map. Episodes start
    at x = -0.75 * size_x.

    Info: ``time``, ``x_distance_travelled`` (m, along x since the start) and ``_light_map``.
    """

    config: LightEscapeEnvConfig

    @classmethod
    def _validate_arena(cls, arena: AquariumArena) -> None:
        if not arena.configuration.sand_ground_color:
            raise ValueError(
                "Light escape requires 'sand_ground_color=True' in the AquariumArenaConfiguration."
            )

    def _cache_references(self, mj_model: mujoco.MjModel) -> None:
        super()._cache_references(mj_model)
        self._arena_size = mj_model.geom("groundplane").size[:2].copy()
        texture = mj_model.texture("groundplane")
        self._texture_id = texture.id
        self._light_map_shape = (int(texture.height[0]), int(texture.width[0]))

        scale = self.config.light_perlin_noise_scale
        if scale and any(size % scale for size in self._light_map_shape):
            raise ValueError(
                f"'light_perlin_noise_scale' ({scale}) must divide both light map dimensions "
                f"{self._light_map_shape}."
            )

        radius, half_length = mj_model.geom_size[self._segment_capsule_geom_ids, :2].T
        self._segment_capsule_areas = np.pi * radius**2 + 2 * radius * 2 * half_length
        disk_radius = mj_model.geom(
            f"{self.morphology_name}/central_disk_pentagon_collider"
        ).size[0]
        self._disk_area = np.pi * disk_radius**2

    def _create_observables(self) -> List[Observable]:
        num_segments = len(self._segment_capsule_geom_ids)
        return super()._create_observables() + [
            Observable(
                name="segment_light_intake",
                low=np.zeros(num_segments),
                high=np.ones(num_segments),
                fn=lambda data, info: self._light_at(
                    info["_light_map"],
                    data.geom_xpos[self._segment_capsule_geom_ids, :2],
                ),
            ),
            Observable(
                name="disk_light_intake",
                low=np.zeros(1),
                high=np.ones(1),
                fn=lambda data, info: self._light_at(
                    info["_light_map"], self._disk_xy(data)[None]
                ),
            ),
        ]

    def _light_at(self, light_map: Any, xy_positions: Any) -> Any:
        """Bilinearly interpolated light at world ``xy_positions`` (N, 2)."""
        xp = self.xp
        size = np.asarray(self._light_map_shape)
        normalized = (xy_positions + self._arena_size) / (2 * self._arena_size)
        # Light map rows run along -y (image convention) and columns along +x.
        coords = xp.stack([1 - normalized[:, 1], normalized[:, 0]], axis=-1) * size

        floor = xp.floor(coords)
        row0 = xp.clip(floor[:, 0].astype(int), 0, size[0] - 1)
        col0 = xp.clip(floor[:, 1].astype(int), 0, size[1] - 1)
        row1 = xp.clip(row0 + 1, 0, size[0] - 1)
        col1 = xp.clip(col0 + 1, 0, size[1] - 1)
        row_frac = coords[:, 0] - row0
        col_frac = coords[:, 1] - col0

        return (
            light_map[row0, col0] * (1 - row_frac) * (1 - col_frac)
            + light_map[row0, col1] * (1 - row_frac) * col_frac
            + light_map[row1, col0] * row_frac * (1 - col_frac)
            + light_map[row1, col1] * row_frac * col_frac
        )

    def _body_light_income(self, observations: Dict[str, Any]) -> Any:
        segments = (
            observations["segment_light_intake"] * self._segment_capsule_areas
        ).sum()
        disk = observations["disk_light_intake"][0] * self._disk_area
        return (segments + disk) / (self._segment_capsule_areas.sum() + self._disk_area)

    def _start_x(self) -> float:
        return -0.75 * float(self._arena_size[0])

    def _initial_disk_position(self) -> np.ndarray:
        return np.array([self._start_x(), 0.0, INITIAL_DISK_HEIGHT])

    def _generate_light_map(self, rng: Any):
        xp = self.xp
        height, width = self._light_map_shape
        light_map = xp.ones((height, width)) * xp.linspace(1, 0, width)

        scale = self.config.light_perlin_noise_scale
        if scale > 0:
            if self.backend.name == "mjc":
                noise_rng = int(rng.integers(0, 10000))
            else:
                import jax

                rng, noise_rng = jax.random.split(rng)
            noise = generate_perlin_noise_2d(
                shape=(height, width), res=(scale, scale), rng_key=noise_rng, npi=xp
            )
            noise = 1 - 2 * noise  # [-1, 1]
            # Noise weights grow along x, leaving the starting zone (first fifth) noise free, so that the light still
            # decreases overall along x.
            start_zone = width // 5
            weights = xp.concatenate(
                [xp.zeros(start_zone), xp.linspace(0, 1, width - start_zone)]
            )
            light_map = light_map * (1 + noise * weights)
            light_map = (light_map - light_map.min()) / (
                light_map.max() - light_map.min()
            )
        return light_map, rng

    def _reset(self, data: Any, rng: Any, **task_options: Any):
        data, rng = self._reset_morphology(data, rng)
        light_map, rng = self._generate_light_map(rng)
        return data, {"_light_map": light_map}, rng

    def _get_info(
        self, state: EnvState, previous_state: Optional[EnvState]
    ) -> Dict[str, Any]:
        return {
            **state.info,
            "time": state.data.time,
            "x_distance_travelled": state.data.xpos[self._disk_body_id, 0]
            - self._start_x(),
        }

    def _get_reward(self, state: EnvState, previous_state: EnvState) -> Any:
        return self._body_light_income(
            previous_state.observations
        ) - self._body_light_income(state.observations)

    def _prepare_render_model(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        observations: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
    ) -> List[int]:
        super()._prepare_render_model(model, data, observations, info)
        # Paint the light map on the sand floor: dark regions are 30 % as bright as fully lit ones.
        shade = 0.3 + 0.7 * np.asarray(info["_light_map"])
        texels = (shade[..., None] * colors.rgba_sand[:3] * 255).astype(np.uint8)
        adr, size = model.tex_adr[self._texture_id], texels.size
        if np.array_equal(model.tex_data[adr : adr + size], texels.ravel()):
            return []
        model.tex_data[adr : adr + size] = texels.ravel()
        return [self._texture_id]


@register("brittle_star/light_escape")
def make_light_escape(
    backend: str = "mjc",
    config: Optional[LightEscapeEnvConfig] = None,
    morphology_specification: Optional[BrittleStarMorphologySpecification] = None,
    arena_configuration: Optional[AquariumArenaConfiguration] = None,
    **config_overrides: Any,
) -> LightEscapeEnv:
    """Light escape in an aquarium with a sand-coloured floor."""
    return make_brittle_star_env(
        env_class=LightEscapeEnv,
        config_class=LightEscapeEnvConfig,
        backend=backend,
        config=config,
        morphology_specification=morphology_specification,
        arena_configuration=arena_configuration,
        default_arena_configuration=partial(
            AquariumArenaConfiguration, sand_ground_color=True
        ),
        config_overrides=config_overrides,
    )
