from itertools import count
from typing import List

import mujoco
import numpy as np
from moojoco.environment.mjc_env import MJCEnvState, MJCObservable
from transforms3d.euler import quat2euler


def get_shared_jumping_spider_mjc_observables(
        mj_model: mujoco.MjModel, mj_data: mujoco.MjData
) -> List[MJCObservable]:
    sensors = [mj_model.sensor(i) for i in range(mj_model.nsensor)]

    # ------------------------------------------------------ Legs ------------------------------------------------------
    leg_joint_pos_sensors = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTPOS and "leg" in sensor.name
    ]
    leg_joints = [mj_model.joint(sensor.objid[0]) for sensor in leg_joint_pos_sensors]

    leg_joint_position_observable = MJCObservable(
        name="leg_joint_position",
        low=np.array([joint.range[0] for joint in leg_joints]),
        high=np.array([joint.range[1] for joint in leg_joints]),
        retriever=lambda state: np.array(
            [
                state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]]
                for sensor in leg_joint_pos_sensors
            ]
        ).flatten(),
    )

    leg_joint_vel_sensors = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTVEL and "leg" in sensor.name
    ]
    leg_joint_velocity_observable = MJCObservable(
        name="leg_joint_velocity",
        low=-np.inf * np.ones(len(leg_joint_vel_sensors)),
        high=np.inf * np.ones(len(leg_joint_vel_sensors)),
        retriever=lambda state: np.array(
            [
                state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]]
                for sensor in leg_joint_vel_sensors
            ]
        ).flatten(),
    )

    # All joint actuator torques
    leg_joint_actuator_frc_sensors = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTACTFRC and "leg" in sensor.name
    ]
    leg_joint_actuator_force_observable = MJCObservable(
        name="leg_joint_actuator_force",
        low=-np.inf * np.ones(len(leg_joint_actuator_frc_sensors)),
        high=np.inf * np.ones(len(leg_joint_actuator_frc_sensors)),
        retriever=lambda state: np.array(
            [
                state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]]
                for sensor in leg_joint_actuator_frc_sensors
            ]
        ).flatten(),
    )

    # All actuator forces
    leg_actuator_frc_sensors = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_ACTUATORFRC and "leg" in sensor.name
    ]
    leg_actuators = [mj_model.actuator(sensor.objid[0]) for sensor in leg_actuator_frc_sensors]
    leg_actuator_force_observable = MJCObservable(
        name="leg_actuator_force",
        low=np.array([actuator.forcerange[0] for actuator in leg_actuators]),
        high=np.array([actuator.forcerange[1] for actuator in leg_actuators]),
        retriever=lambda state: np.array(
            [
                state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]]
                for sensor in leg_actuator_frc_sensors
            ]
        ).flatten(),
    )

    # ----------------------------------------------------- Abdomen ----------------------------------------------------
    abdomen_joint_pos_sensors = [sensor for sensor in sensors if
                                 sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTPOS and "abdomen" in sensor.name]
    abdomen_joints = [mj_model.joint(sensor.objid[0]) for sensor in abdomen_joint_pos_sensors]
    abdomen_joint_pos_observable = MJCObservable(
        name="abdomen_joint_position",
        low=np.array([joint.range[0] for joint in abdomen_joints]),
        high=np.array([joint.range[1] for joint in abdomen_joints]),
        retriever=lambda state: np.array(
            [
                state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]]
                for sensor in abdomen_joint_pos_sensors
            ]
        ).flatten(),
    )

    abdomen_joint_vel_sensors = [sensor for sensor in sensors if
                                 sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTVEL and "abdomen" in sensor.name]
    abdomen_joint_vel_observable = MJCObservable(
        name="abdomen_joint_velocity",
        low=-np.inf * np.ones(len(abdomen_joint_vel_sensors)),
        high=np.inf * np.ones(len(abdomen_joint_vel_sensors)),
        retriever=lambda state: np.array(
            [
                state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]]
                for sensor in abdomen_joint_vel_sensors
            ]
        ).flatten(),
    )

    abdomen_actuatorfrc_sensors = [sensor for sensor in sensors if
                                   sensor.type[0] == mujoco.mjtSensor.mjSENS_ACTUATORFRC and "abdomen" in sensor.name]
    abdomen_actuators = [mj_model.actuator(sensor.objid[0]) for sensor in abdomen_actuatorfrc_sensors]
    abdomen_actuatorfrc_observable = MJCObservable(
        name="abdomen_actuator_force",
        low=np.array([actuator.forcerange[0] for actuator in abdomen_actuators]),
        high=np.array([actuator.forcerange[0] for actuator in abdomen_actuators]),
        retriever=lambda state: np.array([
            state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]]
            for sensor in abdomen_actuatorfrc_sensors])
    )

    abdomen_joint_actuatorfrc_sensors = [sensor for sensor in sensors if
                                         sensor.type[
                                             0] == mujoco.mjtSensor.mjSENS_JOINTACTFRC and "abdomen" in sensor.name]
    abdomen_joint_actuatorfrc_observable = MJCObservable(
        name="abdomen_joint_actuator_force",
        low=-np.inf * np.ones(len(abdomen_joint_actuatorfrc_sensors)),
        high=np.inf * np.ones(len(abdomen_joint_actuatorfrc_sensors)),
        retriever=lambda state: np.array([
            state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]]
            for sensor in abdomen_joint_actuatorfrc_sensors])
    )

    # ------------------------------------------------ Leg tip contacts ------------------------------------------------
    #   Start by mapping geom indices of segment capsules to a contact output index
    indexer = count(0)
    leg_tip_geom_id_to_contact_id = {}
    for geom_id in range(mj_model.ngeom):
        geom_name = mj_model.geom(geom_id).name
        if "_tarsus_" in geom_name and "capsule" in geom_name:
            leg_tip_geom_id_to_contact_id[geom_id] = next(indexer)

    def get_segment_contacts(state: MJCEnvState) -> np.ndarray:
        contacts = np.zeros(len(leg_tip_geom_id_to_contact_id), dtype=int)
        # based on https://gist.github.com/WuXinyang2012/b6649817101dfcb061eff901e9942057
        for contact_id in range(state.mj_data.ncon):
            contact = state.mj_data.contact[contact_id]
            if contact.dist < 0:
                if contact.geom[0] in leg_tip_geom_id_to_contact_id:
                    contacts[leg_tip_geom_id_to_contact_id[contact.geom[0]]] = 1
                if contact.geom[1] in leg_tip_geom_id_to_contact_id:
                    contacts[leg_tip_geom_id_to_contact_id[contact.geom[1]]] = 1

        return contacts

    touch_observable = MJCObservable(
        name="leg_tip_contact",
        low=np.zeros(len(leg_tip_geom_id_to_contact_id)),
        high=np.ones(len(leg_tip_geom_id_to_contact_id)),
        retriever=get_segment_contacts,
    )

    # ------------------------------------------------- Cephalothorax --------------------------------------------------
    framepos_sensor = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMEPOS and "cephalothorax" in sensor.name
    ][0]
    cephalothorax_position_observable = MJCObservable(
        name="cephalothorax_position",
        low=-np.inf * np.ones(3),
        high=np.inf * np.ones(3),
        retriever=lambda state: np.array(
            state.mj_data.sensordata[
            framepos_sensor.adr[0]: framepos_sensor.adr[0] + framepos_sensor.dim[0]
            ]
        ),
    )

    framequat_sensor = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMEQUAT and "cephalothorax" in sensor.name
    ][0]
    cephalothorax_rotation_observable = MJCObservable(
        name="cephalothorax_rotation",
        low=-np.pi * np.ones(3),
        high=np.pi * np.ones(3),
        retriever=lambda state: np.array(
            quat2euler(
                state.mj_data.sensordata[
                framequat_sensor.adr[0]: framequat_sensor.adr[0]
                                         + framequat_sensor.dim[0]
                ]
            )
        ),
    )

    framelinvel_sensor = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMELINVEL and "cephalothorax" in sensor.name
    ][0]
    cephalothorax_linvel_observable = MJCObservable(
        name="cephalothorax_linear_velocity",
        low=-np.inf * np.ones(3),
        high=np.inf * np.ones(3),
        retriever=lambda state: np.array(
            state.mj_data.sensordata[
            framelinvel_sensor.adr[0]: framelinvel_sensor.adr[0]
                                       + framelinvel_sensor.dim[0]
            ]
        ),
    )

    frameangvel_sensor = [
        sensor
        for sensor in sensors
        if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMEANGVEL and "cephalothorax" in sensor.name
    ][0]
    cephalothorax_angvel_observable = MJCObservable(
        name="cephalothorax_angular_velocity",
        low=-np.inf * np.ones(3),
        high=np.inf * np.ones(3),
        retriever=lambda state: np.array(
            state.mj_data.sensordata[
            frameangvel_sensor.adr[0]: frameangvel_sensor.adr[0]
                                       + frameangvel_sensor.dim[0]
            ]
        ),
    )

    observables = [
        leg_joint_position_observable,
        leg_joint_velocity_observable,
        leg_joint_actuator_force_observable,
        leg_actuator_force_observable,
        abdomen_joint_pos_observable,
        abdomen_joint_vel_observable,
        abdomen_joint_actuatorfrc_observable,
        abdomen_actuatorfrc_observable,
        touch_observable,
        cephalothorax_position_observable,
        cephalothorax_rotation_observable,
        cephalothorax_linvel_observable,
        cephalothorax_angvel_observable,
    ]

    # ---------------------------------------------------- Dragline ----------------------------------------------------
    dragline_length_sensors = [sensor for sensor in sensors if
                               sensor.type[0] == mujoco.mjtSensor.mjSENS_TENDONPOS and "dragline" in sensor.name]
    if dragline_length_sensors:
        dragline_length_observable = MJCObservable(
            name="dragline_length",
            low=np.zeros(1),
            high=np.inf * np.ones(1),
            retriever=lambda state: np.array(
                [state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]] for sensor in
                 dragline_length_sensors]).flatten()
        )

        dragline_velocity_sensors = [sensor for sensor in sensors if
                                     sensor.type[0] == mujoco.mjtSensor.mjSENS_TENDONVEL and "dragline" in sensor.name]
        dragline_velocity_observable = MJCObservable(
            name="dragline_velocity",
            low=-np.inf * np.ones(1),
            high=np.inf * np.ones(1),
            retriever=lambda state: np.array(
                [state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]] for sensor in
                 dragline_velocity_sensors]).flatten()
        )

        dragline_actuatorfrc_sensors = [sensor for sensor in sensors if
                                        sensor.type[
                                            0] == mujoco.mjtSensor.mjSENS_ACTUATORFRC and "dragline" in sensor.name]
        dragline_actuator = mj_model.actuator(dragline_actuatorfrc_sensors[0].objid[0])
        dragline_actuatorfrc_observable = MJCObservable(
            name="dragline_actuator_force",
            low=dragline_actuator.forcerange[0] * np.ones(1),
            high=dragline_actuator.forcerange[1] * np.ones(1),
            retriever=lambda state: np.array(
                [state.mj_data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]] for sensor in
                 dragline_actuatorfrc_sensors]).flatten()
        )

        observables += [dragline_length_observable, dragline_velocity_observable, dragline_actuatorfrc_observable]

    return observables
