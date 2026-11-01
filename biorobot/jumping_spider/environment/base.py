"""Shared base of the jumping spider environments: configuration, observations, resets and jump detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import mujoco
import numpy as np

from biorobot.core.backend import Array
from biorobot.core.builder import Arena, compose
from biorobot.core.env import BRTEnv, EnvConfig, EnvState, Observable
from biorobot.core.math import quat_to_euler
from biorobot.core.observables import (
    find_sensors,
    float32_bounds,
    sensordata_indices,
)
from biorobot.jumping_spider import colors
from biorobot.jumping_spider.arenas import JumpingSpiderArena
from biorobot.jumping_spider.morphology import (
    MORPHOLOGY_NAME,
    JumpingSpiderMorphology,
    add_dragline,
)
from biorobot.jumping_spider.specification import (
    JumpingSpiderMorphologySpecification,
    default_jumping_spider_morphology_specification,
)

PREFIX = f"{MORPHOLOGY_NAME}/"
#: Clearance (m) between the spider's lowest point and the ground at reset; it settles onto its legs within ~15 ms.
START_CLEARANCE = 0.0005


@dataclass
class JumpingSpiderEnvConfig(EnvConfig):
    """Settings shared by all jumping spider environments.

    The physics timestep is 1 ms: the gram-scale distal leg segments driven by saturated (bang-bang) actions make the
    simulation diverge at 2 ms.

    Args:
        joint_randomization_noise_scale: Joint angles are initialised uniformly within ``+-scale`` rad of the resting
            pose (clipped to the joint ranges); the spider is lifted so that its lowest point stays above the ground.
        min_flight_time: A flight phase (no contact between the spider and the arena at all) must last at least this
            long (s) to count as a jump.
        color_contacts: Colour the leg tips that touch something when rendering.
    """

    simulation_time: float = 3.0
    physics_timestep: float = 0.001
    num_physics_steps_per_control_step: int = 10
    joint_randomization_noise_scale: float = 0.0
    min_flight_time: float = 0.1
    color_contacts: bool = False
    camera_ids: Sequence[int] = (2,)


class JumpingSpiderEnv(BRTEnv):
    """Base class of the jumping spider tasks.

    ``info`` always contains ``time``, ``airborne`` (no contact with the arena), ``flight_time`` (duration of the
    current flight phase, s) and ``has_jumped`` (a flight phase of at least ``min_flight_time`` happened).
    """

    morphology_name = MORPHOLOGY_NAME
    arena_type: Type[JumpingSpiderArena] = JumpingSpiderArena
    config: JumpingSpiderEnvConfig

    def __init__(
        self, spec: mujoco.MjSpec, config: JumpingSpiderEnvConfig, backend: str = "mjc"
    ) -> None:
        # Number of contacts between the spider and the arena (only those pairs can collide).
        spec.add_sensor(
            name="spider_contacts",
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            intprm=[1, 0, 1],  # data="found", reduce="none", num=1
        )
        self._add_task_sensors(spec)
        super().__init__(spec=spec, config=config, backend=backend)

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: JumpingSpiderMorphology,
        arena: Arena,
        config: JumpingSpiderEnvConfig,
        backend: str = "mjc",
    ) -> "JumpingSpiderEnv":
        if morphology.name != cls.morphology_name:
            raise ValueError(
                f"{cls.__name__} expects a '{cls.morphology_name}' morphology, got '{morphology.name}'."
            )
        cls._validate_arena(arena)
        spec = compose(arena=arena, morphology=morphology, free_joint=True)
        if morphology.specification.dragline.enabled:
            add_dragline(spec, morphology.specification, anchor_site="dragline_anchor")
        return cls(spec=spec, config=config, backend=backend)

    @classmethod
    def _validate_arena(cls, arena: Arena) -> None:
        if not isinstance(arena, cls.arena_type):
            raise TypeError(
                f"{cls.__name__} requires a {cls.arena_type.__name__}, got {type(arena).__name__}."
            )

    def _add_task_sensors(self, spec: mujoco.MjSpec) -> None:
        """Hook to add task-specific sensors to the composed spec before it is compiled."""

    def _configure_options(self, spec: mujoco.MjSpec) -> None:
        super()._configure_options(spec)
        # Contact and constraint buffers per world for MuJoCo Warp (its defaults are too small): every collision
        # capsule touches a surface in at most 2 points, which gives 3 constraint rows each (condim 3, pyramidal
        # friction cone: 4 rows); every hinge can hit a joint limit. Ignored by MuJoCo C and MJX.
        num_capsules = sum(
            geom.name.startswith(PREFIX) and geom.contype != 0 for geom in spec.geoms
        )
        if spec.nconmax < 0:
            spec.nconmax = 2 * num_capsules
        if spec.njmax < 0:
            spec.njmax = 4 * spec.nconmax + len(spec.joints)

    # ------------------------------------------------------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------------------------------------------------------
    def _cache_references(self, mj_model: mujoco.MjModel) -> None:
        free_joint = mj_model.joint(f"{PREFIX}freejoint")
        self._free_qpos_adr = int(free_joint.qposadr[0])
        self._cephalothorax_id = mj_model.body(f"{PREFIX}cephalothorax").id

        joints = [
            j
            for j in range(mj_model.njnt)
            if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            and mj_model.joint(j).name.startswith(PREFIX)
        ]
        self._joint_qpos_adrs = mj_model.jnt_qposadr[joints].astype(int)
        self._joint_ranges = mj_model.jnt_range[joints].copy()

        self._collision_geom_ids = np.array(
            [
                g
                for g in range(mj_model.ngeom)
                if mj_model.geom_contype[g] != 0
                and mj_model.geom(g).name.startswith(PREFIX)
            ],
            dtype=int,
        )
        assert np.all(
            mj_model.geom_type[self._collision_geom_ids]
            == mujoco.mjtGeom.mjGEOM_CAPSULE
        )
        self._collision_geom_radii = mj_model.geom_size[self._collision_geom_ids, 0]
        self._collision_geom_half_lengths = mj_model.geom_size[
            self._collision_geom_ids, 1
        ]
        self._tarsus_geom_ids = np.array(
            [
                g
                for g in range(mj_model.ngeom)
                if mj_model.geom(g).name.endswith("_tarsus")
            ],
            dtype=int,
        )
        self._contact_count_index = sensordata_indices(
            mj_model, [mj_model.sensor("spider_contacts").id]
        )[0]
        self._start_height = self._standing_height(mj_model) + START_CLEARANCE

    def _standing_height(self, mj_model: mujoco.MjModel) -> float:
        """Height of the cephalothorax above the ground when the lowest point of the resting spider touches it."""
        data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, data)
        return float(data.xpos[self._cephalothorax_id, 2] - self._lowest_point(data))

    def _lowest_point(self, data: Any) -> Array:
        """z-coordinate (m) of the lowest point of the spider's collision capsules (requires up-to-date kinematics)."""
        ids = self._collision_geom_ids
        # geom_xmat is (ngeom, 9) in MjData and (ngeom, 3, 3) in MJX; element 8 is the z-component of the local z-axis.
        axis_z = self.xp.reshape(data.geom_xmat[ids], (len(ids), 9))[:, 8]
        bottoms = (
            data.geom_xpos[ids, 2]
            - self.xp.abs(axis_z) * self._collision_geom_half_lengths
            - self._collision_geom_radii
        )
        return self.xp.min(bottoms)

    # ------------------------------------------------------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------------------------------------------------------
    def _sensor_indices(
        self, sensor_type: mujoco.mjtSensor, name_contains: str
    ) -> Tuple[List[int], np.ndarray]:
        sensor_ids = find_sensors(self.mj_model, sensor_type, [PREFIX, name_contains])
        return sensor_ids, sensordata_indices(self.mj_model, sensor_ids)

    def _joint_observables(self, part: str) -> List[Observable]:
        """Joint positions (rad), velocities (rad/s) and actuator forces (N m) of the legs or the abdomen."""
        xp = self.xp
        model = self.mj_model
        pos_ids, pos_indices = self._sensor_indices(
            mujoco.mjtSensor.mjSENS_JOINTPOS, f"/{part}"
        )
        joint_ranges = model.jnt_range[model.sensor_objid[pos_ids]]
        low, high = joint_ranges[:, 0].copy(), joint_ranges[:, 1].copy()
        _, vel_indices = self._sensor_indices(
            mujoco.mjtSensor.mjSENS_JOINTVEL, f"/{part}"
        )
        frc_ids, frc_indices = self._sensor_indices(
            mujoco.mjtSensor.mjSENS_ACTUATORFRC, f"/{part}"
        )
        force_ranges = model.actuator_forcerange[model.sensor_objid[frc_ids]]
        clip_low, clip_high = float32_bounds(low, high)
        inf = np.full(len(vel_indices), np.inf)
        return [
            # Joint limits are soft constraints that can be exceeded by a fraction of a degree; clip to the bounds.
            Observable(
                f"{part}_joint_position",
                low,
                high,
                lambda data, info: xp.clip(
                    data.sensordata[pos_indices], clip_low, clip_high
                ),
            ),
            Observable(
                f"{part}_joint_velocity",
                -inf,
                inf,
                lambda data, info: data.sensordata[vel_indices],
            ),
            Observable(
                f"{part}_actuator_force",
                force_ranges[:, 0].copy(),
                force_ranges[:, 1].copy(),
                lambda data, info: data.sensordata[frc_indices],
            ),
        ]

    def _create_observables(self) -> List[Observable]:
        xp = self.xp
        observables = self._joint_observables("leg") + self._joint_observables(
            "abdomen"
        )

        _, touch_indices = self._sensor_indices(mujoco.mjtSensor.mjSENS_TOUCH, "_touch")
        num_legs = len(touch_indices)
        observables += [
            Observable(
                "leg_tip_contact",
                np.zeros(num_legs),
                np.ones(num_legs),
                lambda data, info: (data.sensordata[touch_indices] > 0).astype(float),
            ),
            Observable(
                "leg_tip_contact_force",
                np.zeros(num_legs),
                np.full(num_legs, np.inf),
                lambda data, info: data.sensordata[touch_indices],
            ),
        ]

        def frame_sensor(sensor_type: mujoco.mjtSensor) -> np.ndarray:
            return self._sensor_indices(sensor_type, "cephalothorax")[1]

        position = frame_sensor(mujoco.mjtSensor.mjSENS_FRAMEPOS)
        quat = frame_sensor(mujoco.mjtSensor.mjSENS_FRAMEQUAT)
        linvel = frame_sensor(mujoco.mjtSensor.mjSENS_FRAMELINVEL)
        angvel = frame_sensor(mujoco.mjtSensor.mjSENS_FRAMEANGVEL)
        inf3 = np.full(3, np.inf)
        observables += [
            Observable(
                "cephalothorax_position",
                -inf3,
                inf3,
                lambda data, info: data.sensordata[position],
            ),
            Observable(
                "cephalothorax_rotation",
                np.array([-np.pi, -np.pi / 2, -np.pi]),
                np.array([np.pi, np.pi / 2, np.pi]),
                lambda data, info: quat_to_euler(data.sensordata[quat], xp),
            ),
            Observable(
                "cephalothorax_linear_velocity",
                -inf3,
                inf3,
                lambda data, info: data.sensordata[linvel],
            ),
            Observable(
                "cephalothorax_angular_velocity",
                -inf3,
                inf3,
                lambda data, info: data.sensordata[angvel],
            ),
        ]

        if self.mj_model.ntendon > 0:
            observables += self._dragline_observables()
        return observables + self._task_observables()

    def _dragline_observables(self) -> List[Observable]:
        _, length = self._sensor_indices(mujoco.mjtSensor.mjSENS_TENDONPOS, "dragline")
        _, velocity = self._sensor_indices(
            mujoco.mjtSensor.mjSENS_TENDONVEL, "dragline"
        )
        force_ids, force = self._sensor_indices(
            mujoco.mjtSensor.mjSENS_ACTUATORFRC, "dragline"
        )
        force_range = self.mj_model.actuator_forcerange[
            self.mj_model.sensor_objid[force_ids]
        ]
        return [
            Observable(
                "dragline_length",
                np.zeros(1),
                np.full(1, np.inf),
                lambda data, info: data.sensordata[length],
            ),
            Observable(
                "dragline_velocity",
                np.full(1, -np.inf),
                np.full(1, np.inf),
                lambda data, info: data.sensordata[velocity],
            ),
            Observable(
                "dragline_brake_force",
                force_range[:, 0].copy(),
                force_range[:, 1].copy(),
                lambda data, info: data.sensordata[force],
            ),
        ]

    def _task_observables(self) -> List[Observable]:
        return []

    # ------------------------------------------------------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------------------------------------------------------
    def _reset_spider(self, data: Any, rng: Any) -> Tuple[Any, Any]:
        """Place the spider in its resting pose at the origin, legs on the ground, with optional joint noise.

        Joint noise changes the spider's lowest point, so the spider is lifted (or lowered) until its lowest point is
        ``START_CLEARANCE`` above the ground, which avoids an impulsive start from interpenetration.
        """
        backend = self.backend
        noise_scale = self.config.joint_randomization_noise_scale
        noise, rng = backend.uniform(
            rng, -noise_scale, noise_scale, shape=(len(self._joint_qpos_adrs),)
        )
        joint_qpos = self.xp.clip(
            noise, self._joint_ranges[:, 0], self._joint_ranges[:, 1]
        )
        height_adr = self._free_qpos_adr + 2
        qpos = backend.at_set(data.qpos, height_adr, self._start_height)
        qpos = backend.at_set(qpos, self._joint_qpos_adrs, joint_qpos)
        data = backend.forward(backend.replace(data, qpos=qpos))
        lift = START_CLEARANCE - self._lowest_point(data)
        qpos = backend.at_set(data.qpos, height_adr, self._start_height + lift)
        return backend.replace(data, qpos=qpos), rng

    def _initial_info(self) -> Dict[str, Array]:
        xp = self.xp
        return {
            "airborne": xp.asarray(False),
            "flight_time": xp.asarray(0.0),
            "has_jumped": xp.asarray(False),
        }

    # ------------------------------------------------------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------------------------------------------------------
    def _get_info(
        self, state: EnvState, previous_state: Optional[EnvState]
    ) -> Dict[str, Array]:
        xp = self.xp
        airborne = state.data.sensordata[self._contact_count_index] == 0
        flight_time = xp.where(
            airborne, state.info["flight_time"] + self.config.control_timestep, 0.0
        )
        if previous_state is None:
            flight_time = xp.zeros_like(flight_time)
        has_jumped = state.info["has_jumped"] | (
            flight_time >= self.config.min_flight_time
        )
        return {
            **state.info,
            "time": state.data.time,
            "airborne": airborne,
            "flight_time": flight_time,
            "has_jumped": has_jumped,
        }

    def _landed(self, state: EnvState) -> Array:
        """True once the spider touches the arena again after a jump."""
        return state.info["has_jumped"] & ~state.info["airborne"]

    # ------------------------------------------------------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------------------------------------------------------
    def _prepare_render_model(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        observations: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
    ) -> List[int]:
        if self.config.color_contacts:
            contacts = observations["leg_tip_contact"] > 0
            model.geom_rgba[self._tarsus_geom_ids] = np.where(
                contacts[:, None], colors.CONTACT, colors.LEG
            )
        return []


def make_morphology(
    specification: Optional[JumpingSpiderMorphologySpecification],
) -> JumpingSpiderMorphology:
    if specification is None:
        specification = default_jumping_spider_morphology_specification()
    return JumpingSpiderMorphology(specification)
