"""Platform jump: jump across a gap of random width from the start platform onto the end platform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from biorobot.core.backend import Array
from biorobot.core.env import EnvState, Observable, make_config
from biorobot.core.observables import sensordata_indices
from biorobot.jumping_spider.arenas import (
    PlatformJumpArena,
    PlatformJumpArenaConfiguration,
)
from biorobot.jumping_spider.environment.base import (
    PREFIX,
    JumpingSpiderEnv,
    JumpingSpiderEnvConfig,
    make_morphology,
)
from biorobot.jumping_spider.specification import JumpingSpiderMorphologySpecification
from biorobot.registry import register


@dataclass
class PlatformJumpEnvConfig(JumpingSpiderEnvConfig):
    """Configuration of the platform jump.

    Args:
        gap_range: The gap between the platforms is sampled uniformly from this range (m) every episode.
        fall_depth: The episode terminates (as a failure) once the cephalothorax is this far (m) below the platforms'
            top.
        success_bonus: Reward for landing with all leg tips on the end platform; it makes a successful jump worth
            more than any approach that falls into the gap.
    """

    gap_range: Tuple[float, float] = (0.1, 0.25)
    fall_depth: float = 0.05
    success_bonus: float = 1.0


class PlatformJumpEnv(JumpingSpiderEnv):
    """Jump from the start platform onto the end platform.

    * Reward: decrease of the distance (m) between the cephalothorax and the centre of the end platform's top surface,
      plus ``success_bonus`` on the step that all leg tips touch the end platform.
    * Terminated: success (all leg tips touch the end platform) or failure (the spider fell into the gap: cephalothorax
      ``fall_depth`` below the platforms).
    * Truncated: time limit.
    * Extra observations: ``unit_direction_to_end_platform`` (3,) and ``distance_to_end_platform`` (1,, m).
    * Reset option: ``gap`` (m) to fix the gap instead of sampling it.
    """

    arena_type = PlatformJumpArena
    config: PlatformJumpEnvConfig

    def _add_task_sensors(self, spec: mujoco.MjSpec) -> None:
        tarsi = [
            geom.name
            for geom in spec.geoms
            if geom.name.startswith(PREFIX) and geom.name.endswith("_tarsus")
        ]
        for tarsus in tarsi:
            spec.add_sensor(
                name=f"{tarsus}_end_platform_contact",
                type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=mujoco.mjtObj.mjOBJ_GEOM,
                objname=tarsus,
                reftype=mujoco.mjtObj.mjOBJ_GEOM,
                refname="end_platform",
                intprm=[1, 0, 1],  # data="found", reduce="none", num=1
            )

    def _cache_references(self, mj_model: mujoco.MjModel) -> None:
        super()._cache_references(mj_model)
        end_platform = mj_model.body("end_platform")
        self._end_platform_mocap_id = int(end_platform.mocapid[0])
        self._end_platform_half_length = float(mj_model.geom("end_platform").size[0])
        start = mj_model.geom("start_platform")
        self._start_platform_edge = float(start.pos[0] + start.size[0])
        self._end_platform_contact_indices = sensordata_indices(
            mj_model,
            [
                s
                for s in range(mj_model.nsensor)
                if mj_model.sensor(s).name.endswith("_end_platform_contact")
            ],
        )

    def _direction_to_end_platform(self, data: Any) -> Array:
        return (
            data.mocap_pos[self._end_platform_mocap_id]
            - data.xpos[self._cephalothorax_id]
        )

    def _task_observables(self) -> List[Observable]:
        xp = self.xp

        def unit_direction(data, info):
            direction = self._direction_to_end_platform(data)
            # Guarded against a zero distance.
            return direction / xp.maximum(xp.linalg.norm(direction), 1e-9)

        return [
            Observable(
                "unit_direction_to_end_platform",
                -np.ones(3),
                np.ones(3),
                unit_direction,
            ),
            Observable(
                "distance_to_end_platform",
                np.zeros(1),
                np.full(1, np.inf),
                lambda data, info: xp.linalg.norm(
                    self._direction_to_end_platform(data)
                ),
            ),
        ]

    def _reset(
        self, data: Any, rng: Any, gap: Optional[float] = None
    ) -> Tuple[Any, Dict[str, Array], Any]:
        data, rng = self._reset_spider(data, rng)
        if gap is None:
            gap, rng = self.backend.uniform(rng, *self.config.gap_range)
        x = self._start_platform_edge + gap + self._end_platform_half_length
        position = self.xp.stack([x, self.xp.zeros_like(x), self.xp.zeros_like(x)])
        mocap_pos = self.backend.at_set(
            data.mocap_pos, self._end_platform_mocap_id, position
        )
        data = self.backend.replace(data, mocap_pos=mocap_pos)
        info = {**self._initial_info(), "gap": self.xp.asarray(gap, dtype=float)}
        return data, info, rng

    def _landed_on_end_platform(self, state: EnvState) -> Array:
        return self.xp.all(
            state.data.sensordata[self._end_platform_contact_indices] > 0
        )

    def _get_reward(self, state: EnvState, previous_state: EnvState) -> Array:
        progress = (
            previous_state.observations["distance_to_end_platform"][0]
            - state.observations["distance_to_end_platform"][0]
        )
        return progress + self.config.success_bonus * self._landed_on_end_platform(
            state
        )

    def _get_terminated(self, state: EnvState) -> Array:
        fell = state.observations["cephalothorax_position"][2] < -self.config.fall_depth
        return self._landed_on_end_platform(state) | fell


@register("jumping_spider/platform_jump")
def make_platform_jump_env(
    backend: str = "mjc",
    config: Optional[PlatformJumpEnvConfig] = None,
    morphology_specification: Optional[JumpingSpiderMorphologySpecification] = None,
    arena_configuration: Optional[PlatformJumpArenaConfiguration] = None,
    **config_overrides: Any,
) -> PlatformJumpEnv:
    return PlatformJumpEnv.from_morphology_and_arena(
        morphology=make_morphology(morphology_specification),
        arena=PlatformJumpArena(
            arena_configuration or PlatformJumpArenaConfiguration()
        ),
        config=make_config(PlatformJumpEnvConfig, config, **config_overrides),
        backend=backend,
    )
