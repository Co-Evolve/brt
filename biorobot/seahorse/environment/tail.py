"""The seahorse tail playground: a tail fixed at its base, driven open-loop through its muscles. There is no task.

The muscle architecture (HM beam spans and routing, MVM beams, position or force control, number of segments, ...) is
configured through the :class:`~biorobot.seahorse.specification.SeahorseMorphologySpecification`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from biorobot.core.backend import Array
from biorobot.core.builder import compose
from biorobot.core.env import BRTEnv, EnvConfig, EnvState, Observable, make_config
from biorobot.core.observables import (
    find_sensors,
    float32_bounds,
    sensordata_indices,
)
from biorobot.registry import register
from biorobot.seahorse.arena import SeahorseArena, SeahorseArenaConfiguration
from biorobot.seahorse.morphology import MORPHOLOGY_NAME, SeahorseMorphology
from biorobot.seahorse.specification import (
    SeahorseMorphologySpecification,
    default_hm_routing_specification,
    default_seahorse_morphology_specification,
)

PREFIX = f"{MORPHOLOGY_NAME}/"


@dataclass
class SeahorseTailEnvConfig(EnvConfig):
    """Configuration of the seahorse tail playground.

    The defaults are those of the original model: a 4 ms physics timestep (MuJoCo's default 2 ms with a time scale of
    2), 40 ms control steps, 5 s episodes, and neither gravity nor contacts.

    Position-controlled HM beams (``p_control=True``; stiff 1000 N/m actuators with forces up to 1000 N) can diverge
    at 4 ms, e.g. long spans under target jumps or ramps that saturate the beams. Use a 1 ms timestep with them
    (``physics_timestep=0.001, num_physics_steps_per_control_step=40``). Force-controlled HM beams and the MVM beams
    are stable at 4 ms under arbitrary actions.

    Args:
        enable_contacts: Simulate contacts between all parts. Note that the (convex hulls of the) nested parts of the
            tail already overlap at rest (121 colliding geom pairs, up to 9.4 mm), as in the original model. Without
            contacts, the geoms' ``contype``/``conaffinity`` are set to 0 as well, so that MJX does not prepare mesh
            collision data.
        enable_gravity: Apply gravity (-z, i.e. along the hanging tail).
    """

    simulation_time: float = 5.0
    physics_timestep: float = 0.004
    num_physics_steps_per_control_step: int = 10
    enable_contacts: bool = False
    enable_gravity: bool = False


class SeahorseTailEnv(BRTEnv):
    """A seahorse tail hanging from a fixed base, actuated by HM and MVM beams.

    * Actions: one per beam actuator (``env.actuator_names``): HM beams first, then MVM beams. Position-controlled beams
      take a target length (m), force-controlled HM beams a value in ``[-1, 0]`` (fraction of the maximum contraction
      force).
    * Reward: always 0. Never terminates; truncated after ``simulation_time``.
    * Observations (empty ones are left out):

      * ``vertebrae_{pitch,roll,yaw}_joint_pos`` (num_segments - 1,): vertebral joint angles (rad), clipped to the
        joint ranges; positive pitch bends the tail ventrally.
      * ``hm_beam_pos`` / ``mvm_beam_pos`` (num_beams,): beam lengths (m).
      * ``actuator_force`` (num_actuators,): actuator forces, <= 0 since beams only pull: the beam force (N) of
        position-controlled beams and the beam force divided by ``f_control_gear`` (i.e. the clipped action) of
        force-controlled ones.
      * ``joint_actuator_force`` (3 * (num_segments - 1),): torque (N m) that the beams exert on every vertebral joint,
        ordered per segment as pitch, roll, yaw.
    """

    morphology_name = MORPHOLOGY_NAME
    config: SeahorseTailEnvConfig

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: SeahorseMorphology,
        arena: SeahorseArena,
        config: SeahorseTailEnvConfig,
        backend: str = "mjc",
    ) -> "SeahorseTailEnv":
        if morphology.name != cls.morphology_name:
            raise ValueError(
                f"{cls.__name__} expects a '{cls.morphology_name}' morphology, got '{morphology.name}'."
            )
        if not isinstance(arena, SeahorseArena):
            raise TypeError(
                f"{cls.__name__} requires a SeahorseArena, got {type(arena).__name__}."
            )
        spec = compose(
            arena=arena,
            morphology=morphology,
            position=arena.configuration.attachment_position,
            euler=arena.configuration.attachment_euler,
            free_joint=False,
        )
        return cls(spec=spec, config=config, backend=backend)

    def _configure_options(self, spec: mujoco.MjSpec) -> None:
        super()._configure_options(spec)
        if not self.config.enable_contacts:
            spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
            for geom in spec.geoms:
                geom.contype = geom.conaffinity = 0
        if not self.config.enable_gravity:
            spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY
        # Contact and constraint buffers per world for MuJoCo Warp (ignored by MuJoCo C and MJX): one row per joint
        # equality, at most a limit and a friction row per joint and, with contacts, 4 rows (condim 3, pyramidal cone)
        # per contact. At rest, the default tail has ~300 contacts in MuJoCo C and ~330 in Warp (~2 per geom).
        if self.config.enable_contacts and spec.nconmax < 0:
            spec.nconmax = 4 * len(spec.geoms)
        if spec.njmax < 0:
            spec.njmax = (
                len(spec.equalities) + 2 * len(spec.joints) + 4 * max(spec.nconmax, 0)
            )

    # ------------------------------------------------------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------------------------------------------------------
    def _sensors(
        self, sensor_type: mujoco.mjtSensor, name_contains: str
    ) -> Tuple[List[int], np.ndarray]:
        sensor_ids = find_sensors(self.mj_model, sensor_type, [PREFIX, name_contains])
        return sensor_ids, sensordata_indices(self.mj_model, sensor_ids)

    def _create_observables(self) -> List[Observable]:
        xp = self.xp
        model = self.mj_model
        observables = []

        for axis in ("pitch", "roll", "yaw"):
            ids, indices = self._sensors(
                mujoco.mjtSensor.mjSENS_JOINTPOS, f"_vertebrae_joint_{axis}_sensor"
            )
            joint_ranges = model.jnt_range[model.sensor_objid[ids]]
            low, high = joint_ranges[:, 0].copy(), joint_ranges[:, 1].copy()
            clip_low, clip_high = float32_bounds(low, high)
            observables.append(
                Observable(
                    f"vertebrae_{axis}_joint_pos",
                    low,
                    high,
                    # Joint limits are soft constraints that can be exceeded slightly; clip to the bounds.
                    lambda data, info, indices=indices, low=clip_low, high=clip_high: xp.clip(
                        data.sensordata[indices], low, high
                    ),
                )
            )

        for muscle in ("hm", "mvm"):
            _, indices = self._sensors(
                mujoco.mjtSensor.mjSENS_TENDONPOS, f"/{muscle}_beam_"
            )
            observables.append(
                Observable(
                    f"{muscle}_beam_pos",
                    np.zeros(len(indices)),
                    np.full(len(indices), np.inf),
                    lambda data, info, indices=indices: data.sensordata[indices],
                )
            )

        ids, force_indices = self._sensors(
            mujoco.mjtSensor.mjSENS_ACTUATORFRC, "_beam_"
        )
        force_range = model.actuator_forcerange[model.sensor_objid[ids]]
        observables.append(
            Observable(
                "actuator_force",
                force_range[:, 0].copy(),
                force_range[:, 1].copy(),
                lambda data, info: data.sensordata[force_indices],
            )
        )

        _, torque_indices = self._sensors(
            mujoco.mjtSensor.mjSENS_JOINTACTFRC, "_vertebrae_joint_"
        )
        inf = np.full(len(torque_indices), np.inf)
        observables.append(
            Observable(
                "joint_actuator_force",
                -inf,
                inf,
                lambda data, info: data.sensordata[torque_indices],
            )
        )
        return [observable for observable in observables if observable.shape[0] > 0]

    # ------------------------------------------------------------------------------------------------------------------
    # Task hooks
    # ------------------------------------------------------------------------------------------------------------------
    def _reset(self, data: Any, rng: Any) -> Tuple[Any, Dict[str, Array], Any]:
        return data, {}, rng

    def _get_reward(self, state: EnvState, previous_state: EnvState) -> Array:
        return 0.0


def default_tail_morphology_specification(
    p_control: bool = False, mvm_enabled: bool = True
) -> SeahorseMorphologySpecification:
    """The prototype's 11-segment tail with a pair of ventral HM beams that span the whole tail (the routing of the
    paper's HM experiment), force-controlled unless ``p_control``, plus MVM beams if ``mvm_enabled``.
    """
    specification = default_seahorse_morphology_specification(
        num_segments=11,
        hm_segment_span=10,
        p_control=p_control,
        mvm_enabled=mvm_enabled,
    )
    specification.beam_actuation_specification.hm_beam_actuation_specification.routing_specifications = [
        default_hm_routing_specification(num_segments=11, segment_span=10)
    ]
    return specification


@register("seahorse/tail")
def make_tail_env(
    backend: str = "mjc",
    config: Optional[SeahorseTailEnvConfig] = None,
    morphology_specification: Optional[SeahorseMorphologySpecification] = None,
    arena_configuration: Optional[SeahorseArenaConfiguration] = None,
    **config_overrides: Any,
) -> SeahorseTailEnv:
    return SeahorseTailEnv.from_morphology_and_arena(
        morphology=SeahorseMorphology(
            morphology_specification or default_tail_morphology_specification()
        ),
        arena=SeahorseArena(arena_configuration or SeahorseArenaConfiguration()),
        config=make_config(SeahorseTailEnvConfig, config, **config_overrides),
        backend=backend,
    )
