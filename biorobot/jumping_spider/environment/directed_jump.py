"""Directed jump: jump towards a target at a random distance and elevation, like a salticid jumping onto prey."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from biorobot.core.backend import Array
from biorobot.core.env import EnvState, Observable, make_config
from biorobot.jumping_spider.arenas import (
    DirectedJumpArena,
    DirectedJumpArenaConfiguration,
)
from biorobot.jumping_spider.environment.base import (
    JumpingSpiderEnv,
    JumpingSpiderEnvConfig,
    make_morphology,
)
from biorobot.jumping_spider.specification import JumpingSpiderMorphologySpecification
from biorobot.registry import register


@dataclass
class DirectedJumpEnvConfig(JumpingSpiderEnvConfig):
    """Configuration of the directed jump.

    The target is placed at ``distance`` from the cephalothorax's start position, in the direction given by
    ``elevation`` (angle above the horizontal) and ``azimuth`` (angle from +x towards +y), each sampled uniformly from
    its range every episode.

    Args:
        target_distance_range: Range of the target distance (m).
        target_elevation_range: Range of the target elevation (rad).
        target_azimuth_range: Range of the target azimuth (rad).
        success_bonus: Reward for reaching the target after a jump.
    """

    target_distance_range: Tuple[float, float] = (0.2, 0.4)
    target_elevation_range: Tuple[float, float] = (0.0, np.pi / 6)
    target_azimuth_range: Tuple[float, float] = (0.0, 0.0)
    success_bonus: float = 1.0


class DirectedJumpEnv(JumpingSpiderEnv):
    """Jump towards a target.

    * Reward: decrease of the distance (m) between the cephalothorax and the target, plus ``success_bonus`` on success.
    * Terminated: success, i.e. the cephalothorax's centre is within the target sphere after a jump (``has_jumped``).
      Low targets can also be reached by walking, which earns the distance reward but neither the bonus nor
      termination.
    * Truncated: time limit.
    * Extra observations: ``unit_direction_to_target`` (3,) and ``distance_to_target`` (1,, m).
    * Reset option: ``target_position`` (3,) to place the target instead of sampling it.
    """

    arena_type = DirectedJumpArena
    config: DirectedJumpEnvConfig

    def _cache_references(self, mj_model: mujoco.MjModel) -> None:
        super()._cache_references(mj_model)
        self._target_mocap_id = int(mj_model.body("target").mocapid[0])
        self._target_radius = float(mj_model.geom("target").size[0])

    def _direction_to_target(self, data: Any) -> Array:
        return data.mocap_pos[self._target_mocap_id] - data.xpos[self._cephalothorax_id]

    def _task_observables(self) -> List[Observable]:
        xp = self.xp

        def unit_direction(data, info):
            direction = self._direction_to_target(data)
            # Guarded against a zero distance.
            return direction / xp.maximum(xp.linalg.norm(direction), 1e-9)

        return [
            Observable(
                "unit_direction_to_target", -np.ones(3), np.ones(3), unit_direction
            ),
            Observable(
                "distance_to_target",
                np.zeros(1),
                np.full(1, np.inf),
                lambda data, info: xp.linalg.norm(self._direction_to_target(data)),
            ),
        ]

    def _reset(
        self,
        data: Any,
        rng: Any,
        target_position: Optional[Sequence[float]] = None,
    ) -> Tuple[Any, Dict[str, Array], Any]:
        xp = self.xp
        data, rng = self._reset_spider(data, rng)
        if target_position is None:
            config = self.config
            distance, rng = self.backend.uniform(rng, *config.target_distance_range)
            elevation, rng = self.backend.uniform(rng, *config.target_elevation_range)
            azimuth, rng = self.backend.uniform(rng, *config.target_azimuth_range)
            offset = distance * xp.stack(
                [
                    xp.cos(elevation) * xp.cos(azimuth),
                    xp.cos(elevation) * xp.sin(azimuth),
                    xp.sin(elevation),
                ]
            )
            target_position = offset + xp.array([0.0, 0.0, self._start_height])
        target_position = xp.asarray(target_position, dtype=float)
        mocap_pos = self.backend.at_set(
            data.mocap_pos, self._target_mocap_id, target_position
        )
        data = self.backend.replace(data, mocap_pos=mocap_pos)
        return data, {**self._initial_info(), "target_position": target_position}, rng

    def _reached_target(self, state: EnvState) -> Array:
        within = state.observations["distance_to_target"][0] < self._target_radius
        return within & state.info["has_jumped"]

    def _get_reward(self, state: EnvState, previous_state: EnvState) -> Array:
        progress = (
            previous_state.observations["distance_to_target"][0]
            - state.observations["distance_to_target"][0]
        )
        return progress + self.config.success_bonus * self._reached_target(state)

    def _get_terminated(self, state: EnvState) -> Array:
        return self._reached_target(state)


@register("jumping_spider/directed_jump")
def make_directed_jump_env(
    backend: str = "mjc",
    config: Optional[DirectedJumpEnvConfig] = None,
    morphology_specification: Optional[JumpingSpiderMorphologySpecification] = None,
    arena_configuration: Optional[DirectedJumpArenaConfiguration] = None,
    **config_overrides: Any,
) -> DirectedJumpEnv:
    return DirectedJumpEnv.from_morphology_and_arena(
        morphology=make_morphology(morphology_specification),
        arena=DirectedJumpArena(
            arena_configuration or DirectedJumpArenaConfiguration()
        ),
        config=make_config(DirectedJumpEnvConfig, config, **config_overrides),
        backend=backend,
    )
