"""Undirected locomotion: move the central disk as far away from the origin as possible, in any direction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from biorobot.brittle_star.arena import AquariumArenaConfiguration
from biorobot.brittle_star.environment.base import (
    BrittleStarEnv,
    BrittleStarEnvConfig,
    make_brittle_star_env,
)
from biorobot.brittle_star.specification import BrittleStarMorphologySpecification
from biorobot.core import EnvState
from biorobot.registry import register


@dataclass
class UndirectedLocomotionEnvConfig(BrittleStarEnvConfig):
    """No task-specific options; see :class:`BrittleStarEnvConfig`."""


class UndirectedLocomotionEnv(BrittleStarEnv):
    """Reward: the increase in xy distance (m) between the central disk and the world origin during the step.

    Info: ``time`` and ``xy_distance_from_origin``.
    """

    config: UndirectedLocomotionEnvConfig

    def _reset(self, data: Any, rng: Any, **task_options: Any):
        data, rng = self._reset_morphology(data, rng)
        return data, {}, rng

    def _get_info(
        self, state: EnvState, previous_state: Optional[EnvState]
    ) -> Dict[str, Any]:
        return {
            **state.info,
            "time": state.data.time,
            "xy_distance_from_origin": self.xp.linalg.norm(self._disk_xy(state.data)),
        }

    def _get_reward(self, state: EnvState, previous_state: EnvState) -> Any:
        return (
            state.info["xy_distance_from_origin"]
            - previous_state.info["xy_distance_from_origin"]
        )


@register("brittle_star/undirected_locomotion")
def make_undirected_locomotion(
    backend: str = "mjc",
    config: Optional[UndirectedLocomotionEnvConfig] = None,
    morphology_specification: Optional[BrittleStarMorphologySpecification] = None,
    arena_configuration: Optional[AquariumArenaConfiguration] = None,
    **config_overrides: Any,
) -> UndirectedLocomotionEnv:
    """Undirected locomotion in the default aquarium (checkered floor, no target)."""
    return make_brittle_star_env(
        env_class=UndirectedLocomotionEnv,
        config_class=UndirectedLocomotionEnvConfig,
        backend=backend,
        config=config,
        morphology_specification=morphology_specification,
        arena_configuration=arena_configuration,
        default_arena_configuration=AquariumArenaConfiguration,
        config_overrides=config_overrides,
    )
