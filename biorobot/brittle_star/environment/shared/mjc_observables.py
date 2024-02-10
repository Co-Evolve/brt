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

    # All actuator torques
    actuator_frc_sensors = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTACTFRC
    ]
    actuator_force_observable = MJCObservable(
        name="joint_actuator_force",
        low=-np.inf * np.ones(len(actuator_frc_sensors)),
        high=np.inf * np.ones(len(actuator_frc_sensors)),
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

    #   Get non-morphology geom ids to check contacts with
    external_contact_geom_ids = {
        mj_model.geom("groundplane").id,
        mj_model.geom("north_wall").id,
        mj_model.geom("east_wall").id,
        mj_model.geom("south_wall").id,
        mj_model.geom("west_wall").id,
    }

    def get_segment_contacts(state: MJCEnvState) -> np.ndarray:
        contacts = np.zeros(len(segment_capsule_geom_id_to_contact_idx))
        # based on https://gist.github.com/WuXinyang2012/b6649817101dfcb061eff901e9942057
        for contact_id in range(state.mj_data.ncon):
            contact = state.mj_data.contact[contact_id]
            if contact.geom1 in external_contact_geom_ids:
                if contact.geom2 in segment_capsule_geom_id_to_contact_idx:
                    c_array = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(
                        m=state.mj_model, d=state.mj_data, id=contact_id, result=c_array
                    )

                    # Convert the contact force from contact frame to world frame
                    ref = np.reshape(contact.frame, (3, 3))
                    c_force = np.dot(np.linalg.inv(ref), c_array[0:3])

                    index = segment_capsule_geom_id_to_contact_idx[contact.geom2]
                    contacts[index] = max(np.linalg.norm(c_force), contacts[index])

        ground_contacts = (contacts > 0).astype(int)
        return ground_contacts

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
        actuator_force_observable,
        touch_observable,
        disk_position_observable,
        disk_rotation_observable,
        disk_linvel_observable,
        disk_angvel_observable,
    ]
