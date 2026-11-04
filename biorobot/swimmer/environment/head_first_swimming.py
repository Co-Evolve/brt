"""Head-first swimming: move the swimmer's head as far as possible along its initial heading (-x)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from biorobot.core import (
    BRTEnv,
    EnvConfig,
    EnvState,
    Observable,
    find_sensors,
    float32_bounds,
    make_config,
    sensordata_indices,
)
from biorobot.core.backend import Array
from biorobot.core.math import quat_to_euler
from biorobot.core.observables import sensor_observable
from biorobot.registry import register
from biorobot.swimmer.arena import FluidArena, FluidArenaConfiguration
from biorobot.swimmer.morphology import MORPHOLOGY_NAME, SwimmerMorphology
from biorobot.swimmer.specification import (
    SwimmerMorphologySpecification,
    default_swimmer_morphology_specification,
)

PREFIX = f"{MORPHOLOGY_NAME}/"


@dataclass
class HeadFirstSwimmingEnvConfig(EnvConfig):
    """Configuration of the head-first swimming task.

    The defaults are a 2 ms timestep (Euler integration) and a new action every physics step, which is
    what a CPG integrated alongside the simulation needs. With ``num_physics_steps_per_control_step > 1``, every
    action is held for that many physics steps and ``actuator_work`` becomes an estimate (see
    :class:`HeadFirstSwimmingEnv`).
    """

    simulation_time: float = 10.0
    num_physics_steps_per_control_step: int = 1
    physics_timestep: float = 0.002


class HeadFirstSwimmingEnv(BRTEnv):
    """The swimmer starts straight along +x with its head at the origin and should swim head first, i.e. along -x.

    * Actions: one target angle (rad) per joint servo, head to tail (``env.actuator_names``), within the joint's
      range.
    * Reward: the head's displacement (m) along -x during the step, so tail-first swimming is penalised. Never
      terminates; truncated after ``simulation_time``.
    * Info: ``time``; ``head_first_distance``, the head's total displacement (m) along -x since the reset; and
      ``actuator_work`` (J), the work the servos did, with negative work counted as cost too: the sum over control
      steps of ``sum_joints |torque * joint velocity| * control_timestep``, both read after the step. With one physics
      step per control step (the default), it integrates over every physics step.
    * Observations:

      * ``joint_position`` (num_joints,): joint angles (rad), clipped to the joint ranges.
      * ``joint_velocity`` (num_joints,): joint angular velocities (rad/s).
      * ``joint_actuator_force`` (num_joints,): servo torque (N m) on every joint.
      * ``head_position`` (3,) and ``head_rotation`` (3,; extrinsic xyz Euler angles, rad) of the head segment.
      * ``head_linear_velocity`` (3,) and ``head_angular_velocity`` (3,) of the head segment.
    """

    morphology_name = MORPHOLOGY_NAME
    config: HeadFirstSwimmingEnvConfig

    @classmethod
    def _validate_arena(cls, arena) -> None:
        if not isinstance(arena, FluidArena):
            raise TypeError(
                f"{cls.__name__} requires a FluidArena, got {type(arena).__name__}."
            )

    def _configure_options(self, spec: mujoco.MjSpec) -> None:
        super()._configure_options(spec)
        # Contact and constraint buffers per world for MuJoCo Warp (ignored by MuJoCo C and MJX). Only capsules that
        # are not neighbours can touch, when the body coils: at most 2 contacts per pair, 4 constraint rows per contact
        # (condim 3, pyramidal cone), plus a limit row per joint.
        num_segments = sum("segment_" in body.name for body in spec.bodies)
        if spec.nconmax < 0:
            spec.nconmax = max(num_segments * (num_segments - 1), 1)
        if spec.njmax < 0:
            spec.njmax = 4 * spec.nconmax + num_segments

    def _cache_references(self, mj_model: mujoco.MjModel) -> None:
        self._head_body_id = mj_model.body(f"{PREFIX}segment_0").id
        joints = [
            i
            for i in range(mj_model.njnt)
            if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self._joint_dof_adrs = mj_model.jnt_dofadr[joints].astype(int)

    # ------------------------------------------------------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------------------------------------------------------
    def _create_observables(self) -> List[Observable]:
        xp, m, s = self.xp, self.mj_model, mujoco.mjtSensor
        joint_ids = find_sensors(m, s.mjSENS_JOINTPOS)
        joint_indices = sensordata_indices(m, joint_ids)
        joint_ranges = m.jnt_range[m.sensor_objid[joint_ids]]
        low, high = joint_ranges[:, 0].copy(), joint_ranges[:, 1].copy()
        clip_low, clip_high = float32_bounds(low, high)
        quat_indices = sensordata_indices(
            m, find_sensors(m, s.mjSENS_FRAMEQUAT, "head")
        )
        return [
            Observable(
                "joint_position",
                low,
                high,
                # Joint limits are soft constraints that can be exceeded slightly; clip to the bounds.
                lambda data, info: xp.clip(
                    data.sensordata[joint_indices], clip_low, clip_high
                ),
            ),
            sensor_observable("joint_velocity", m, s.mjSENS_JOINTVEL),
            sensor_observable("joint_actuator_force", m, s.mjSENS_JOINTACTFRC),
            sensor_observable("head_position", m, s.mjSENS_FRAMEPOS, "head"),
            Observable(
                "head_rotation",
                np.full(3, -np.pi),
                np.full(3, np.pi),
                lambda data, info: quat_to_euler(data.sensordata[quat_indices], xp),
            ),
            sensor_observable("head_linear_velocity", m, s.mjSENS_FRAMELINVEL, "head"),
            sensor_observable("head_angular_velocity", m, s.mjSENS_FRAMEANGVEL, "head"),
        ]

    # ------------------------------------------------------------------------------------------------------------------
    # Task hooks
    # ------------------------------------------------------------------------------------------------------------------
    def _reset(self, data: Any, rng: Any) -> Tuple[Any, Dict[str, Array], Any]:
        xp = self.xp
        return data, {"actuator_work": xp.asarray(0.0)}, rng

    def _get_info(
        self, state: EnvState, previous_state: Optional[EnvState]
    ) -> Dict[str, Array]:
        xp, data = self.xp, state.data
        info = {
            **state.info,
            "time": data.time,
            "head_first_distance": -data.xpos[self._head_body_id, 0],
        }
        if previous_state is not None:
            adrs = self._joint_dof_adrs
            power = xp.sum(xp.abs(data.qfrc_actuator[adrs] * data.qvel[adrs]))
            info["actuator_work"] = (
                previous_state.info["actuator_work"]
                + power * self.config.control_timestep
            )
        return info

    def _get_reward(self, state: EnvState, previous_state: EnvState) -> Array:
        return (
            state.info["head_first_distance"]
            - previous_state.info["head_first_distance"]
        )


@register("swimmer/head_first_swimming")
def make_head_first_swimming_env(
    backend: str = "mjc",
    config: Optional[HeadFirstSwimmingEnvConfig] = None,
    morphology_specification: Optional[SwimmerMorphologySpecification] = None,
    arena_configuration: Optional[FluidArenaConfiguration] = None,
    **config_overrides: Any,
) -> HeadFirstSwimmingEnv:
    """Head-first swimming with the default 8-segment swimmer in the default fluid.

    The default swimmer uses the ellipsoid fluid model, which ``backend="mjx"`` does not support; use ``"mjc"`` or
    ``"warp"``, or a specification with ``fluid_model="inertia"``.
    """
    return HeadFirstSwimmingEnv.from_morphology_and_arena(
        morphology=SwimmerMorphology(
            morphology_specification or default_swimmer_morphology_specification()
        ),
        arena=FluidArena(arena_configuration or FluidArenaConfiguration()),
        config=make_config(HeadFirstSwimmingEnvConfig, config, **config_overrides),
        backend=backend,
    )
