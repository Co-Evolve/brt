"""Directed locomotion: move the central disk towards a target."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np

from biorobot.brittle_star.arena import AquariumArena, AquariumArenaConfiguration
from biorobot.brittle_star.environment.base import (
    BrittleStarEnv,
    BrittleStarEnvConfig,
    make_brittle_star_env,
)
from biorobot.brittle_star.specification import BrittleStarMorphologySpecification
from biorobot.core import EnvState, Observable
from biorobot.registry import register

#: Height (m) of randomly placed targets.
TARGET_HEIGHT = 0.05


@dataclass
class DirectedLocomotionEnvConfig(BrittleStarEnvConfig):
    """Args:
    target_distance: xy distance (m) from the origin of randomly placed targets (at a uniformly random angle).
    """

    target_distance: float = 3.0


class DirectedLocomotionEnv(BrittleStarEnv):
    """Reward: the decrease in xy distance (m) between the central disk and the target during the step.

    The episode terminates once the disk is within the target's radius (0.2 m). Pass ``target_position`` (xyz) to
    ``reset`` to place the target yourself; a target position containing NaNs is replaced by a random one (handy to mix
    both within one vmapped batch).

    Info: ``time`` and ``xy_target_position``.
    """

    config: DirectedLocomotionEnvConfig

    @classmethod
    def _validate_arena(cls, arena: AquariumArena) -> None:
        if not arena.configuration.attach_target:
            raise ValueError(
                "Directed locomotion requires a target: set 'attach_target=True' in the AquariumArenaConfiguration."
            )

    def _cache_references(self, mj_model: mujoco.MjModel) -> None:
        super()._cache_references(mj_model)
        target = mj_model.body("target")
        self._target_mocap_id = int(mj_model.body_mocapid[target.id])
        self._target_radius = float(mj_model.site("target_site").size[0])

    def _create_observables(self) -> List[Observable]:
        return super()._create_observables() + [
            Observable(
                name="unit_xy_direction_to_target",
                low=-np.ones(2),
                high=np.ones(2),
                # Guarded against a zero distance (disk exactly on the target).
                fn=lambda data, info: self._xy_direction_to_target(data)
                / self.xp.maximum(self._xy_distance_to_target(data), 1e-9),
            ),
            Observable(
                name="xy_distance_to_target",
                low=np.zeros(1),
                high=np.full(1, np.inf),
                fn=lambda data, info: self._xy_distance_to_target(data),
            ),
        ]

    def _xy_direction_to_target(self, data: Any) -> Any:
        return data.mocap_pos[self._target_mocap_id, :2] - self._disk_xy(data)

    def _xy_distance_to_target(self, data: Any) -> Any:
        return self.xp.linalg.norm(self._xy_direction_to_target(data))

    def _reset(self, data: Any, rng: Any, target_position: Any = None):
        xp = self.xp
        angle, rng = self.backend.uniform(rng, 0.0, 2 * np.pi)
        distance = self.config.target_distance
        random_position = xp.stack(
            [
                distance * xp.cos(angle),
                distance * xp.sin(angle),
                xp.asarray(TARGET_HEIGHT),
            ]
        )
        if target_position is None:
            position = random_position
        else:
            target_position = xp.asarray(target_position, dtype=float)
            position = xp.where(
                xp.any(xp.isnan(target_position)), random_position, target_position
            )
        mocap_pos = self.backend.at_set(data.mocap_pos, self._target_mocap_id, position)
        data = self.backend.replace(data, mocap_pos=mocap_pos)
        data, rng = self._reset_morphology(data, rng)
        return data, {}, rng

    def _get_info(
        self, state: EnvState, previous_state: Optional[EnvState]
    ) -> Dict[str, Any]:
        return {
            **state.info,
            "time": state.data.time,
            # Fancy indexing copies: with "mjc", a slice would be a view on the mocap positions, updated in place.
            "xy_target_position": state.data.mocap_pos[
                self._target_mocap_id, np.arange(2)
            ],
        }

    def _get_reward(self, state: EnvState, previous_state: EnvState) -> Any:
        return (
            previous_state.observations["xy_distance_to_target"][0]
            - state.observations["xy_distance_to_target"][0]
        )

    def _get_terminated(self, state: EnvState) -> Any:
        return state.observations["xy_distance_to_target"][0] < self._target_radius


@register("brittle_star/directed_locomotion")
def make_directed_locomotion(
    backend: str = "mjc",
    config: Optional[DirectedLocomotionEnvConfig] = None,
    morphology_specification: Optional[BrittleStarMorphologySpecification] = None,
    arena_configuration: Optional[AquariumArenaConfiguration] = None,
    **config_overrides: Any,
) -> DirectedLocomotionEnv:
    """Directed locomotion in an aquarium with a target."""
    return make_brittle_star_env(
        env_class=DirectedLocomotionEnv,
        config_class=DirectedLocomotionEnvConfig,
        backend=backend,
        config=config,
        morphology_specification=morphology_specification,
        arena_configuration=arena_configuration,
        default_arena_configuration=partial(
            AquariumArenaConfiguration, attach_target=True
        ),
        config_overrides=config_overrides,
    )
