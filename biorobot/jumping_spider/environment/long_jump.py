"""Long jump: jump as far forward (+x) as possible."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from biorobot.core.backend import Array
from biorobot.core.env import EnvState, Observable, make_config
from biorobot.jumping_spider.arenas import LongJumpArena, LongJumpArenaConfiguration
from biorobot.jumping_spider.environment.base import (
    JumpingSpiderEnv,
    JumpingSpiderEnvConfig,
    make_morphology,
)
from biorobot.jumping_spider.specification import JumpingSpiderMorphologySpecification
from biorobot.registry import register


@dataclass
class LongJumpEnvConfig(JumpingSpiderEnvConfig):
    """Configuration of the long jump.

    Args:
        terminate_on_landing: End the episode when the spider lands after its first jump (a flight phase of at least
            ``min_flight_time``), so that the return equals the distance covered up to the landing.
    """

    terminate_on_landing: bool = True


class LongJumpEnv(JumpingSpiderEnv):
    """Maximise the forward distance of the cephalothorax.

    * Reward: forward (x) displacement of the cephalothorax during the step (m).
    * Terminated: the spider lands after a jump (if ``terminate_on_landing``), or the cephalothorax leaves the track
      sideways (a failure).
    * Truncated: time limit.
    * Extra observation ``x_distance`` (1,): x-coordinate of the cephalothorax (m); the spider starts at x = 0.
    """

    arena_type = LongJumpArena
    config: LongJumpEnvConfig

    def _cache_references(self, mj_model) -> None:
        super()._cache_references(mj_model)
        self._track_half_width = float(mj_model.geom("track").size[1])

    def _task_observables(self) -> List[Observable]:
        return [
            Observable(
                "x_distance",
                np.full(1, -np.inf),
                np.full(1, np.inf),
                lambda data, info: data.xpos[self._cephalothorax_id, :1],
            )
        ]

    def _reset(self, data: Any, rng: Any) -> Tuple[Any, Dict[str, Array], Any]:
        data, rng = self._reset_spider(data, rng)
        return data, self._initial_info(), rng

    def _get_reward(self, state: EnvState, previous_state: EnvState) -> Array:
        return (
            state.observations["x_distance"][0]
            - previous_state.observations["x_distance"][0]
        )

    def _get_terminated(self, state: EnvState) -> Array:
        off_track = (
            self.xp.abs(state.observations["cephalothorax_position"][1])
            > self._track_half_width
        )
        return (self._landed(state) & self.config.terminate_on_landing) | off_track


@register("jumping_spider/long_jump")
def make_long_jump_env(
    backend: str = "mjc",
    config: Optional[LongJumpEnvConfig] = None,
    morphology_specification: Optional[JumpingSpiderMorphologySpecification] = None,
    arena_configuration: Optional[LongJumpArenaConfiguration] = None,
    **config_overrides: Any,
) -> LongJumpEnv:
    return LongJumpEnv.from_morphology_and_arena(
        morphology=make_morphology(morphology_specification),
        arena=LongJumpArena(arena_configuration or LongJumpArenaConfiguration()),
        config=make_config(LongJumpEnvConfig, config, **config_overrides),
        backend=backend,
    )
