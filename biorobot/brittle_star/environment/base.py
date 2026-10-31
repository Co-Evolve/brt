"""Shared configuration, observables, reset and rendering logic of all brittle star tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import mujoco
import numpy as np

from biorobot.brittle_star.arena import AquariumArena, AquariumArenaConfiguration
from biorobot.brittle_star.morphology import BrittleStarMorphology
from biorobot.brittle_star.specification import (
    BrittleStarMorphologySpecification,
    default_brittle_star_morphology_specification,
)
from biorobot.core import (
    BRTEnv,
    EnvConfig,
    Observable,
    find_sensors,
    make_config,
    sensordata_indices,
)
from biorobot.core.math import quat_to_euler, yaw_to_quat
from biorobot.core.observables import sensor_observable
from biorobot.utils import colors

#: Initial height of the central disk (m); the brittle star drops onto the floor at the start of an episode.
INITIAL_DISK_HEIGHT = 0.11


@dataclass
class BrittleStarEnvConfig(EnvConfig):
    """Settings shared by all brittle star tasks (see :class:`~biorobot.core.EnvConfig` for the generic ones).

    Args:
        joint_randomization_noise_scale: Initial segment joint positions (rad) and velocities (rad/s) are sampled
            uniformly from ``[-scale, scale]``.
        color_contacts: When rendering, colour segments that touch something red.
        random_initial_rotation: Start every episode with a uniformly random rotation around the world z-axis.

    The solver defaults (1 iteration, 5 line-search iterations, no Euler damping) trade accuracy for speed, which
    matters most with MJX; they match the settings the environments were tuned with.
    """

    joint_randomization_noise_scale: float = 0.0
    color_contacts: bool = False
    random_initial_rotation: bool = False
    solver_iterations: Optional[int] = 1
    solver_ls_iterations: Optional[int] = 5
    disable_eulerdamp: bool = True


class BrittleStarEnv(BRTEnv):
    """Base class of the brittle star tasks: observables, initial state and contact colouring."""

    morphology_name = "BrittleStarMorphology"
    config: BrittleStarEnvConfig

    def _configure_options(self, spec: mujoco.MjSpec) -> None:
        super()._configure_options(spec)
        # Contact and constraint buffers per world for MuJoCo Warp (its defaults are too small for many segments): a
        # capsule touches a plane in at most 2 points and the disk cylinder in at most 4 (+ 4 for walls), a condim 4
        # (pyramidal) contact has 6 constraint rows, and each segment can hit 2 joint limits. Ignored by MuJoCo C/MJX.
        num_segments = sum("_segment_" in body.name for body in spec.bodies)
        if spec.nconmax < 0:
            spec.nconmax = 2 * num_segments + 8
        if spec.njmax < 0:
            spec.njmax = 6 * spec.nconmax + 2 * num_segments

    def _cache_references(self, mj_model: mujoco.MjModel) -> None:
        prefix = f"{self.morphology_name}/"
        self._disk_body_id = mj_model.body(f"{prefix}central_disk").id
        self._free_qpos_adr = int(mj_model.joint(f"{prefix}freejoint").qposadr[0])

        segment_joints = [
            i
            for i in range(mj_model.njnt)
            if "segment" in mj_model.joint(i).name
            and mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self._segment_joint_qpos_adrs = mj_model.jnt_qposadr[segment_joints].astype(int)
        self._segment_joint_qvel_adrs = mj_model.jnt_dofadr[segment_joints].astype(int)
        self._segment_capsule_geom_ids = np.array(
            [
                i
                for i in range(mj_model.ngeom)
                if "segment" in mj_model.geom(i).name
                and "capsule" in mj_model.geom(i).name
            ],
            dtype=int,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------------------------------------------------------
    def _create_observables(self) -> List[Observable]:
        m, s = self.mj_model, mujoco.mjtSensor
        joints = [m.sensor_objid[i] for i in find_sensors(m, s.mjSENS_JOINTPOS)]
        # Extrinsic xyz Euler angles (rad) of the disk.
        quat_indices = sensordata_indices(
            m, find_sensors(m, s.mjSENS_FRAMEQUAT, "disk")
        )
        return [
            sensor_observable(
                "joint_position",
                m,
                s.mjSENS_JOINTPOS,
                low=m.jnt_range[joints, 0],
                high=m.jnt_range[joints, 1],
            ),
            sensor_observable("joint_velocity", m, s.mjSENS_JOINTVEL),
            sensor_observable("joint_actuator_force", m, s.mjSENS_JOINTACTFRC),
            sensor_observable(
                "actuator_force",
                m,
                s.mjSENS_ACTUATORFRC,
                low=m.actuator_forcerange[:, 0],
                high=m.actuator_forcerange[:, 1],
            ),
            sensor_observable("disk_position", m, s.mjSENS_FRAMEPOS, "disk"),
            Observable(
                name="disk_rotation",
                low=np.full(3, -np.pi),
                high=np.full(3, np.pi),
                fn=lambda data, info: quat_to_euler(
                    data.sensordata[quat_indices], self.xp
                ),
            ),
            sensor_observable("disk_linear_velocity", m, s.mjSENS_FRAMELINVEL, "disk"),
            sensor_observable("disk_angular_velocity", m, s.mjSENS_FRAMEANGVEL, "disk"),
            sensor_observable("tendon_position", m, s.mjSENS_TENDONPOS, low=0.0),
            sensor_observable("tendon_velocity", m, s.mjSENS_TENDONVEL),
            sensor_observable("segment_contact", m, s.mjSENS_TOUCH, low=0.0),
        ]

    # ------------------------------------------------------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------------------------------------------------------
    def _initial_disk_position(self) -> np.ndarray:
        return np.array([0.0, 0.0, INITIAL_DISK_HEIGHT])

    def _reset_morphology(self, data: Any, rng: Any) -> Tuple[Any, Any]:
        """Place the disk, optionally rotate it around z, and add joint noise. Returns ``(data, rng)``."""
        backend, xp, config = self.backend, self.xp, self.config
        adr = self._free_qpos_adr
        qpos = backend.at_set(
            data.qpos, slice(adr, adr + 3), self._initial_disk_position()
        )

        if config.random_initial_rotation:
            yaw, rng = backend.uniform(rng, -np.pi, np.pi)
            qpos = backend.at_set(qpos, slice(adr + 3, adr + 7), yaw_to_quat(yaw, xp))

        scale = config.joint_randomization_noise_scale
        num_joints = len(self._segment_joint_qpos_adrs)
        qpos_noise, rng = backend.uniform(rng, -scale, scale, shape=(num_joints,))
        qvel_noise, rng = backend.uniform(rng, -scale, scale, shape=(num_joints,))
        qpos = backend.at_set(
            qpos,
            self._segment_joint_qpos_adrs,
            self.mj_model.qpos0[self._segment_joint_qpos_adrs] + qpos_noise,
        )
        qvel = backend.at_set(data.qvel, self._segment_joint_qvel_adrs, qvel_noise)
        return backend.replace(data, qpos=qpos, qvel=qvel), rng

    # ------------------------------------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------------------------------------
    def _disk_xy(self, data: Any) -> Any:
        return data.xpos[self._disk_body_id, :2]

    def _prepare_render_model(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        observations: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
    ) -> List[int]:
        if self.config.color_contacts:
            contacts = observations["segment_contact"].reshape(
                len(self._segment_capsule_geom_ids), -1
            )
            in_contact = np.any(contacts > 0, axis=-1)
            model.geom_rgba[self._segment_capsule_geom_ids] = np.where(
                in_contact[:, None], colors.rgba_red, colors.rgba_green
            )
        return []


def make_brittle_star_env(
    env_class: Type[BrittleStarEnv],
    config_class: Type[BrittleStarEnvConfig],
    backend: str,
    config: Optional[BrittleStarEnvConfig],
    morphology_specification: Optional[BrittleStarMorphologySpecification],
    arena_configuration: Optional[AquariumArenaConfiguration],
    default_arena_configuration: Callable[[], AquariumArenaConfiguration],
    config_overrides: Dict[str, Any],
) -> BrittleStarEnv:
    """Shared body of the registered factories: build the defaults that were not given and the environment."""
    config = make_config(config_class, config, **config_overrides)
    if morphology_specification is None:
        morphology_specification = default_brittle_star_morphology_specification(
            num_arms=5,
            num_segments_per_arm=5,
            use_p_control=True,
            radius_to_strength_factor=200,
        )
    arena = AquariumArena(arena_configuration or default_arena_configuration())
    return env_class.from_morphology_and_arena(
        morphology=BrittleStarMorphology(morphology_specification),
        arena=arena,
        config=config,
        backend=backend,
    )
