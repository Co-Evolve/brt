from itertools import count
from typing import List, Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax.scipy.spatial.transform import Rotation
from moojoco.environment.base import BaseObservable, BaseEnvState
from moojoco.environment.mjc_env import MJCObservable, MJCEnvState
from moojoco.environment.mjx_env import MJXObservable, MJXEnvState
from transforms3d.euler import quat2euler


def get_quat2euler_fn(backend: str) -> Callable[[chex.Array], chex.Array]:
    if backend == "mjx":

        def jquat2euler(quat):
            xywz = jnp.roll(a=quat, shift=-1)
            return Rotation.from_quat(xywz).as_euler(seq="xyz", degrees=False)

        return jquat2euler
    else:
        return quat2euler


def get_num_contacts_and_segment_contacts_fn(
    mj_model: mujoco.MjModel, backend: str
) -> Tuple[int, Callable[[BaseEnvState], chex.Array]]:
    if backend == "mjx":
        segment_capsule_geom_ids = np.array(
            [
                geom_id
                for geom_id in range(mj_model.ngeom)
                if "segment" in mj_model.geom(geom_id).name
                and "capsule" in mj_model.geom(geom_id).name
            ]
        )

        def get_segment_contacts(state: MJXEnvState) -> jnp.ndarray:
            contact_data = state.mjx_data.contact
            contacts = contact_data.dist <= 0

            def solve_contact(geom_id: int) -> jnp.ndarray:
                return (
                    jnp.sum(contacts * jnp.any(geom_id == contact_data.geom, axis=-1))
                    > 0
                ).astype(int)

            return jax.vmap(solve_contact)(segment_capsule_geom_ids)

        num_contacts = len(segment_capsule_geom_ids)
    else:
        # segment touch values
        #   Start by mapping geom indices of segment capsules to a contact output index
        indexer = count(0)
        segment_capsule_geom_id_to_contact_idx = {}
        for geom_id in range(mj_model.ngeom):
            geom_name = mj_model.geom(geom_id).name
            if "segment" in geom_name and "capsule" in geom_name:
                segment_capsule_geom_id_to_contact_idx[geom_id] = next(indexer)

        def get_segment_contacts(state: MJCEnvState) -> np.ndarray:
            contacts = np.zeros(len(segment_capsule_geom_id_to_contact_idx), dtype=int)
            # based on https://gist.github.com/WuXinyang2012/b6649817101dfcb061eff901e9942057
            for contact_id in range(state.mj_data.ncon):
                contact = state.mj_data.contact[contact_id]
                if contact.dist < 0:
                    if contact.geom1 in segment_capsule_geom_id_to_contact_idx:
                        contacts[
                            segment_capsule_geom_id_to_contact_idx[contact.geom1]
                        ] = 1
                    if contact.geom2 in segment_capsule_geom_id_to_contact_idx:
                        contacts[
                            segment_capsule_geom_id_to_contact_idx[contact.geom2]
                        ] = 1

            return contacts

        num_contacts = len(segment_capsule_geom_id_to_contact_idx)
    return num_contacts, get_segment_contacts


def get_base_brittle_star_observables(
    mj_model: mujoco.MjModel, backend: str
) -> List[BaseObservable]:
    if backend == "mjx":
        observable_class = MJXObservable
        bnp = jnp
        get_data = lambda state: state.mjx_data
    else:
        observable_class = MJCObservable
        bnp = np
        get_data = lambda state: state.mj_data

    sensors = [mj_model.sensor(i) for i in range(mj_model.nsensor)]

    # All joint positions
    joint_pos_sensors = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTPOS
    ]
    joints = [mj_model.joint(sensor.objid[0]) for sensor in joint_pos_sensors]

    joint_position_observable = observable_class(
        name="joint_position",
        low=bnp.array([joint.range[0] for joint in joints]),
        high=bnp.array([joint.range[1] for joint in joints]),
        retriever=lambda state: bnp.array(
            [
                get_data(state).sensordata[
                    sensor.adr[0] : sensor.adr[0] + sensor.dim[0]
                ]
                for sensor in joint_pos_sensors
            ]
        ).flatten(),
    )

    # All joint velocities
    joint_vel_sensors = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTVEL
    ]
    joint_velocity_observable = observable_class(
        name="joint_velocity",
        low=-bnp.inf * bnp.ones(len(joint_vel_sensors)),
        high=bnp.inf * bnp.ones(len(joint_vel_sensors)),
        retriever=lambda state: bnp.array(
            [
                get_data(state).sensordata[
                    sensor.adr[0] : sensor.adr[0] + sensor.dim[0]
                ]
                for sensor in joint_vel_sensors
            ]
        ).flatten(),
    )

    # All joint actuator torques
    joint_actuator_frc_sensors = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTACTFRC
    ]
    joint_actuator_force_observable = observable_class(
        name="joint_actuator_force",
        low=-bnp.inf * bnp.ones(len(joint_actuator_frc_sensors)),
        high=bnp.inf * bnp.ones(len(joint_actuator_frc_sensors)),
        retriever=lambda state: bnp.array(
            [
                get_data(state).sensordata[
                    sensor.adr[0] : sensor.adr[0] + sensor.dim[0]
                ]
                for sensor in joint_actuator_frc_sensors
            ]
        ).flatten(),
    )

    # All actuator forces
    actuator_frc_sensors = [
        mj_model.sensor(i)
        for i in range(mj_model.nsensor)
        if mj_model.sensor(i).type[0] == mujoco.mjtSensor.mjSENS_ACTUATORFRC
    ]
    actuator_force_observable = observable_class(
        name="actuator_force",
        low=bnp.array([limits[0] for limits in mj_model.actuator_forcerange]),
        high=bnp.array([limits[1] for limits in mj_model.actuator_forcerange]),
        retriever=lambda state: bnp.array(
            [
                get_data(state).sensordata[
                    sensor.adr[0] : sensor.adr[0] + sensor.dim[0]
                ]
                for sensor in actuator_frc_sensors
            ]
        ).flatten(),
    )

    # disk pos
    disk_framepos_sensor = [
        mj_model.sensor(i)
        for i in range(mj_model.nsensor)
        if mj_model.sensor(i).type[0] == mujoco.mjtSensor.mjSENS_FRAMEPOS
        and "disk" in mj_model.sensor(i).name
    ][0]
    disk_position_observable = observable_class(
        name="disk_position",
        low=-bnp.inf * bnp.ones(3),
        high=bnp.inf * bnp.ones(3),
        retriever=lambda state: get_data(state).sensordata[
            disk_framepos_sensor.adr[0] : disk_framepos_sensor.adr[0]
            + disk_framepos_sensor.dim[0]
        ],
    )
    # disk rotation
    disk_framequat_sensor = [
        mj_model.sensor(i)
        for i in range(mj_model.nsensor)
        if mj_model.sensor(i).type[0] == mujoco.mjtSensor.mjSENS_FRAMEQUAT
        and "disk" in mj_model.sensor(i).name
    ][0]
    disk_rotation_observable = observable_class(
        name="disk_rotation",
        low=-bnp.pi * bnp.ones(3),
        high=bnp.pi * bnp.ones(3),
        retriever=lambda state: get_quat2euler_fn(backend=backend)(
            get_data(state).sensordata[
                disk_framequat_sensor.adr[0] : disk_framequat_sensor.adr[0]
                + disk_framequat_sensor.dim[0]
            ]
        ),
    )

    # disk linvel
    disk_framelinvel_sensor = [
        mj_model.sensor(i)
        for i in range(mj_model.nsensor)
        if mj_model.sensor(i).type[0] == mujoco.mjtSensor.mjSENS_FRAMELINVEL
        and "disk" in mj_model.sensor(i).name
    ][0]
    disk_linvel_observable = observable_class(
        name="disk_linear_velocity",
        low=-bnp.inf * bnp.ones(3),
        high=bnp.inf * bnp.ones(3),
        retriever=lambda state: get_data(state).sensordata[
            disk_framelinvel_sensor.adr[0] : disk_framelinvel_sensor.adr[0]
            + disk_framelinvel_sensor.dim[0]
        ],
    )

    # disk angvel
    disk_frameangvel_sensor = [
        mj_model.sensor(i)
        for i in range(mj_model.nsensor)
        if mj_model.sensor(i).type[0] == mujoco.mjtSensor.mjSENS_FRAMEANGVEL
        and "disk" in mj_model.sensor(i).name
    ][0]
    disk_angvel_observable = observable_class(
        name="disk_angular_velocity",
        low=-bnp.inf * bnp.ones(3),
        high=bnp.inf * bnp.ones(3),
        retriever=lambda state: get_data(state).sensordata[
            disk_frameangvel_sensor.adr[0] : disk_frameangvel_sensor.adr[0]
            + disk_frameangvel_sensor.dim[0]
        ],
    )

    # tendons
    tendon_pos_sensors = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_TENDONPOS
    ]
    tendon_pos_observable = observable_class(
        name="tendon_position",
        low=-bnp.inf * bnp.ones(len(tendon_pos_sensors)),
        high=bnp.inf * bnp.ones(len(tendon_pos_sensors)),
        retriever=lambda state: bnp.array(
            [
                get_data(state).sensordata[
                    sensor.adr[0] : sensor.adr[0] + sensor.dim[0]
                ]
                for sensor in tendon_pos_sensors
            ]
        ).flatten(),
    )

    tendon_vel_sensors = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_TENDONVEL
    ]
    tendon_vel_observable = observable_class(
        name="tendon_velocity",
        low=-bnp.inf * bnp.ones(len(tendon_vel_sensors)),
        high=bnp.inf * bnp.ones(len(tendon_vel_sensors)),
        retriever=lambda state: bnp.array(
            [
                get_data(state).sensordata[
                    sensor.adr[0] : sensor.adr[0] + sensor.dim[0]
                ]
                for sensor in tendon_vel_sensors
            ]
        ).flatten(),
    )

    # contacts
    num_contacts, get_segment_contacts_fn = get_num_contacts_and_segment_contacts_fn(
        mj_model=mj_model, backend=backend
    )
    segment_contact_observable = observable_class(
        name="segment_contact",
        low=np.zeros(num_contacts),
        high=np.ones(num_contacts),
        retriever=get_segment_contacts_fn,
    )

    return [
        joint_position_observable,
        joint_velocity_observable,
        joint_actuator_force_observable,
        actuator_force_observable,
        disk_position_observable,
        disk_rotation_observable,
        disk_linvel_observable,
        disk_angvel_observable,
        tendon_pos_observable,
        tendon_vel_observable,
        segment_contact_observable,
    ]
