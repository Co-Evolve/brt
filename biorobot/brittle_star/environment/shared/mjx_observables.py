from typing import List

import jax
import jax.numpy as jnp
import mujoco
from jax._src.scipy.spatial.transform import Rotation
from moojoco.environment.mjx_env import MJXEnvState, MJXObservable

"""
This will be merged with the MJC observables once MJX supports sensors.
"""


def get_shared_brittle_star_mjx_observables(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData
) -> List[MJXObservable]:
    joints = [
        mj_model.joint(joint_id)
        for joint_id in range(mj_model.njnt)
        if "segment" in mj_model.joint(joint_id).name
    ]
    joints_qpos_adr = jnp.array([joint.qposadr[0] for joint in joints])
    joint_dof_adr = jnp.array([joint.dofadr[0] for joint in joints])
    joint_range = jnp.array([joint.range for joint in joints]).T

    joint_position_observable = MJXObservable(
        name="joint_position",
        low=joint_range[0],
        high=joint_range[1],
        retriever=lambda state: state.mjx_data.qpos[joints_qpos_adr],
    )

    joint_velocity_observable = MJXObservable(
        name="joint_velocity",
        low=-jnp.inf * jnp.ones(len(joints)),
        high=jnp.inf * jnp.ones(len(joints)),
        retriever=lambda state: state.mjx_data.qvel[joint_dof_adr],
    )

    joint_actuator_force_observable = MJXObservable(
        name="joint_actuator_force",
        low=-jnp.inf * jnp.ones(len(joint_dof_adr)),
        high=jnp.inf * jnp.ones(len(joint_dof_adr)),
        retriever=lambda state: state.mjx_data.qfrc_actuator[joint_dof_adr].flatten(),
    )

    low_actuator_force_limit = jnp.array(
        [limits[0] for limits in mj_model.actuator_forcerange]
    )
    high_actuator_force_limit = jnp.array(
        [limits[1] for limits in mj_model.actuator_forcerange]
    )

    def calculate_actuator_force(state: MJXEnvState) -> jnp.ndarray:
        l = state.mjx_data.actuator_length
        v = state.mjx_data.actuator_velocity
        gain = state.mjx_model.actuator_gainprm[:, 0]
        b0 = state.mjx_model.actuator_biasprm[:, 0]
        b1 = state.mjx_model.actuator_biasprm[:, 1]
        b2 = state.mjx_model.actuator_biasprm[:, 2]
        ctrl = state.mjx_data.ctrl

        return jnp.clip(
            gain * ctrl + b0 + b1 * l + b2 * v,
            low_actuator_force_limit,
            high_actuator_force_limit,
        )

    actuator_force_observable = MJXObservable(
        name="actuator_force",
        low=low_actuator_force_limit,
        high=high_actuator_force_limit,
        retriever=calculate_actuator_force,
    )

    segment_capsule_geom_ids = jnp.array(
        [
            geom_id
            for geom_id in range(mj_model.ngeom)
            if "segment" in mj_model.geom(geom_id).name
            and "capsule" in mj_model.geom(geom_id).name
        ]
    )

    external_contact_geom_ids = jnp.array(
        [
            mj_model.geom("groundplane").id,
            mj_model.geom("north_wall").id,
            mj_model.geom("east_wall").id,
            mj_model.geom("south_wall").id,
            mj_model.geom("west_wall").id,
        ]
    )

    def get_segment_contacts(state: MJXEnvState) -> jnp.ndarray:
        contact_data = state.mjx_data.contact
        contacts = contact_data.dist <= 0

        def solve_contact(geom_id: int) -> jnp.ndarray:
            return (
                jnp.sum(contacts * jnp.any(geom_id == contact_data.geom, axis=-1)) > 0
            ).astype(int)

        return jax.vmap(solve_contact)(segment_capsule_geom_ids)

    touch_observable = MJXObservable(
        name="segment_contact",
        low=jnp.zeros(len(segment_capsule_geom_ids)),
        high=jnp.ones(len(segment_capsule_geom_ids)),
        retriever=get_segment_contacts,
    )

    disk_id = mj_model.body("BrittleStarMorphology/central_disk").id
    disk_position_observable = MJXObservable(
        name="disk_position",
        low=-jnp.inf * jnp.ones(3),
        high=jnp.inf * jnp.ones(3),
        retriever=lambda state: state.mjx_data.xpos[disk_id],
    )

    # disk framequat
    disk_rotation_observable = MJXObservable(
        name="disk_rotation",
        low=-jnp.pi * jnp.ones(3),
        high=jnp.pi * jnp.ones(3),
        retriever=lambda state: Rotation.from_quat(
            quat=state.mjx_data.xquat[disk_id]
        ).as_euler(seq="xyz"),
    )

    # disk framelinvel
    morphology_freejoint_adr = mj_model.joint(
        "BrittleStarMorphology/freejoint/"
    ).dofadr[0]
    disk_linvel_observable = MJXObservable(
        name="disk_linear_velocity",
        low=-jnp.inf * jnp.ones(3),
        high=jnp.inf * jnp.ones(3),
        retriever=lambda state: state.mjx_data.qvel[
            morphology_freejoint_adr : morphology_freejoint_adr + 3
        ],
    )
    # disk frameangvel
    disk_angvel_observable = MJXObservable(
        name="disk_angular_velocity",
        low=-jnp.inf * jnp.ones(3),
        high=jnp.inf * jnp.ones(3),
        retriever=lambda state: state.mjx_data.qvel[
            morphology_freejoint_adr + 3 : morphology_freejoint_adr + 6
        ],
    )

    return [
        joint_position_observable,
        joint_velocity_observable,
        joint_actuator_force_observable,
        actuator_force_observable,
        touch_observable,
        disk_position_observable,
        disk_rotation_observable,
        disk_linvel_observable,
        disk_angvel_observable,
    ]
