from itertools import count
from typing import List

import mujoco
import numpy as np
from moojoco.environment.mjc_env import MJCEnvState, MJCObservable
from transforms3d.euler import quat2euler


def get_shared_brittle_star_mjc_observables(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData
) -> List[MJCObservable]:
    sensors = [mj_model.sensor(i) for i in range(mj_model.nsensor)]

    # All joint positions
    joint_pos_sensors = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTPOS
    ]
    joints = [mj_model.joint(sensor.objid[0]) for sensor in joint_pos_sensors]

    joint_position_observable = MJCObservable(
        name="joint_position",
        low=np.array([joint.range[0] for joint in joints]),
        high=np.array([joint.range[1] for joint in joints]),
        retriever=lambda state: np.array(
            [
                state.mj_data.sensordata[sensor.adr[0] : sensor.adr[0] + sensor.dim[0]]
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
    joint_velocity_observable = MJCObservable(
        name="joint_velocity",
        low=-np.inf * np.ones(len(joint_vel_sensors)),
        high=np.inf * np.ones(len(joint_vel_sensors)),
        retriever=lambda state: np.array(
            [
                state.mj_data.sensordata[sensor.adr[0] : sensor.adr[0] + sensor.dim[0]]
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
    joint_actuator_force_observable = MJCObservable(
        name="joint_actuator_force",
        low=-np.inf * np.ones(len(joint_actuator_frc_sensors)),
        high=np.inf * np.ones(len(joint_actuator_frc_sensors)),
        retriever=lambda state: np.array(
            [
                state.mj_data.sensordata[sensor.adr[0] : sensor.adr[0] + sensor.dim[0]]
                for sensor in joint_actuator_frc_sensors
            ]
        ).flatten(),
    )

    # All actuator forces
    actuator_frc_sensors = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_ACTUATORFRC
    ]
    actuator_force_observable = MJCObservable(
        name="actuator_force",
        low=np.array([limits[0] for limits in mj_model.actuator_forcerange]),
        high=np.array([limits[1] for limits in mj_model.actuator_forcerange]),
        retriever=lambda state: np.array(
            [
                state.mj_data.sensordata[sensor.adr[0] : sensor.adr[0] + sensor.dim[0]]
                for sensor in actuator_frc_sensors
            ]
        ).flatten(),
    )

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
                    contacts[segment_capsule_geom_id_to_contact_idx[contact.geom1]] = 1
                if contact.geom2 in segment_capsule_geom_id_to_contact_idx:
                    contacts[segment_capsule_geom_id_to_contact_idx[contact.geom2]] = 1

        return contacts

    touch_observable = MJCObservable(
        name="segment_contact",
        low=np.zeros(len(segment_capsule_geom_id_to_contact_idx)),
        high=np.ones(len(segment_capsule_geom_id_to_contact_idx)),
        retriever=get_segment_contacts,
    )

    # disk framepos
    framepos_sensor = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMEPOS
    ][0]
    disk_position_observable = MJCObservable(
        name="disk_position",
        low=-np.inf * np.ones(3),
        high=np.inf * np.ones(3),
        retriever=lambda state: np.array(
            state.mj_data.sensordata[
                framepos_sensor.adr[0] : framepos_sensor.adr[0] + framepos_sensor.dim[0]
            ]
        ),
    )

    # disk framequat
    framequat_sensor = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMEQUAT
    ][0]
    disk_rotation_observable = MJCObservable(
        name="disk_rotation",
        low=-np.pi * np.ones(3),
        high=np.pi * np.ones(3),
        retriever=lambda state: np.array(
            quat2euler(
                state.mj_data.sensordata[
                    framequat_sensor.adr[0] : framequat_sensor.adr[0]
                    + framequat_sensor.dim[0]
                ]
            )
        ),
    )

    # disk framelinvel
    framelinvel_sensor = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMELINVEL
    ][0]
    disk_linvel_observable = MJCObservable(
        name="disk_linear_velocity",
        low=-np.inf * np.ones(3),
        high=np.inf * np.ones(3),
        retriever=lambda state: np.array(
            state.mj_data.sensordata[
                framelinvel_sensor.adr[0] : framelinvel_sensor.adr[0]
                + framelinvel_sensor.dim[0]
            ]
        ),
    )

    # disk frameangvel
    frameangvel_sensor = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMEANGVEL
    ][0]
    disk_angvel_observable = MJCObservable(
        name="disk_angular_velocity",
        low=-np.inf * np.ones(3),
        high=np.inf * np.ones(3),
        retriever=lambda state: np.array(
            state.mj_data.sensordata[
                frameangvel_sensor.adr[0] : frameangvel_sensor.adr[0]
                + frameangvel_sensor.dim[0]
            ]
        ),
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
