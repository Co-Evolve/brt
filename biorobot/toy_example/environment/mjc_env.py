from itertools import count
from typing import Any, Dict, List

import mujoco
import numpy as np
from moojoco.environment.base import MuJoCoEnvironmentConfiguration
from moojoco.environment.mjc_env import MJCEnv, MJCEnvState, MJCObservable
from transforms3d.euler import quat2euler

from biorobot.toy_example.mjcf.arena.arena import MJCFPlaneWithTargetArena
from biorobot.toy_example.mjcf.morphology.morphology import MJCFToyExampleMorphology


class ToyExampleEnvironmentConfiguration(MuJoCoEnvironmentConfiguration):
    def __init__(
        self,
        target_distance: float = 3,
        joint_randomization_noise_scale: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_distance = target_distance
        self.joint_randomization_noise_scale = joint_randomization_noise_scale


class ToyExampleMJCEnvironment(MJCEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        configuration: ToyExampleEnvironmentConfiguration,
        mjcf_str: str,
        mjcf_assets: Dict[str, Any],
    ) -> None:
        super().__init__(
            mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration
        )

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: MJCFToyExampleMorphology,
        arena: MJCFPlaneWithTargetArena,
        configuration: ToyExampleEnvironmentConfiguration,
    ) -> MJCEnv:
        return super().from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=configuration
        )

    @property
    def environment_configuration(self) -> ToyExampleEnvironmentConfiguration:
        return super().environment_configuration

    @staticmethod
    def _get_xy_direction_to_target(state: MJCEnvState) -> np.ndarray:
        target_position = state.mj_data.body("target").xpos
        torso_position = state.mj_data.body("ToyExampleMorphology/torso").xpos
        direction_to_target = target_position - torso_position
        return direction_to_target[:2]

    @staticmethod
    def _get_xy_distance_to_target(state: MJCEnvState) -> float:
        xy_direction_to_target = ToyExampleMJCEnvironment._get_xy_direction_to_target(
            state=state
        )
        xy_distance_to_target = np.linalg.norm(xy_direction_to_target)
        return xy_distance_to_target

    def _create_observables(self) -> List[MJCObservable]:
        sensors = [
            self.frozen_mj_model.sensor(i) for i in range(self.frozen_mj_model.nsensor)
        ]

        # All joint positions
        joint_pos_sensors = [
            sensor
            for sensor in sensors
            if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTPOS
        ]
        in_plane_joint_pos_sensors = [
            sensor for sensor in joint_pos_sensors if "in_plane_joint" in sensor.name
        ]
        in_plane_joints = [
            self.frozen_mj_model.joint(sensor.objid[0])
            for sensor in in_plane_joint_pos_sensors
        ]
        out_of_plane_joint_pos_sensors = [
            sensor
            for sensor in joint_pos_sensors
            if "out_of_plane_joint" in sensor.name
        ]
        out_of_plane_joints = [
            self.frozen_mj_model.joint(sensor.objid[0])
            for sensor in out_of_plane_joint_pos_sensors
        ]

        in_plane_joint_pos_observable = MJCObservable(
            name="in_plane_joint_position",
            low=np.array([joint.range[0] for joint in in_plane_joints]),
            high=np.array([joint.range[1] for joint in in_plane_joints]),
            retriever=lambda state: np.array(
                [
                    state.mj_data.sensordata[
                        sensor.adr[0] : sensor.adr[0] + sensor.dim[0]
                    ]
                    for sensor in in_plane_joint_pos_sensors
                ]
            ).flatten(),
        )
        out_of_plane_joint_pos_observable = MJCObservable(
            name="out_of_plane_joint_position",
            low=np.array([joint.range[0] for joint in out_of_plane_joints]),
            high=np.array([joint.range[1] for joint in out_of_plane_joints]),
            retriever=lambda state: np.array(
                [
                    state.mj_data.sensordata[
                        sensor.adr[0] : sensor.adr[0] + sensor.dim[0]
                    ]
                    for sensor in out_of_plane_joint_pos_sensors
                ]
            ).flatten(),
        )

        # All joint velocities
        joint_vel_sensors = [
            sensor
            for sensor in sensors
            if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTVEL
        ]
        in_plane_joint_velocity_sensors = [
            sensor for sensor in joint_vel_sensors if "in_plane_joint" in sensor.name
        ]
        out_of_plane_joint_velocity_sensors = [
            sensor
            for sensor in joint_vel_sensors
            if "out_of_plane_joint" in sensor.name
        ]
        in_plane_joint_velocity_observable = MJCObservable(
            name="in_plane_joint_velocity",
            low=-np.inf * np.ones(len(in_plane_joint_velocity_sensors)),
            high=np.inf * np.ones(len(in_plane_joint_velocity_sensors)),
            retriever=lambda state: np.array(
                [
                    state.mj_data.sensordata[
                        sensor.adr[0] : sensor.adr[0] + sensor.dim[0]
                    ]
                    for sensor in in_plane_joint_velocity_sensors
                ]
            ).flatten(),
        )
        out_of_plane_joint_velocity_observable = MJCObservable(
            name="out_of_plane_joint_velocity",
            low=-np.inf * np.ones(len(out_of_plane_joint_velocity_sensors)),
            high=np.inf * np.ones(len(out_of_plane_joint_velocity_sensors)),
            retriever=lambda state: np.array(
                [
                    state.mj_data.sensordata[
                        sensor.adr[0] : sensor.adr[0] + sensor.dim[0]
                    ]
                    for sensor in out_of_plane_joint_velocity_sensors
                ]
            ).flatten(),
        )

        # segment touch values
        indexer = count(0)
        segment_capsule_geom_ids_to_contact_idx = {
            geom_id: next(indexer)
            for geom_id in range(self.frozen_mj_model.ngeom)
            if "segment" in self.frozen_mj_model.geom(geom_id).name
            and "capsule" in self.frozen_mj_model.geom(geom_id).name
        }
        ground_floor_geom_id = self.frozen_mj_model.geom("groundplane").id

        def get_segment_ground_contacts(state: MJCEnvState) -> np.ndarray:
            ground_contacts = np.zeros(len(segment_capsule_geom_ids_to_contact_idx))
            # based on https://gist.github.com/WuXinyang2012/b6649817101dfcb061eff901e9942057
            for contact_id in range(state.mj_data.ncon):
                contact = state.mj_data.contact[contact_id]
                if contact.geom1 == ground_floor_geom_id:
                    if contact.geom2 in segment_capsule_geom_ids_to_contact_idx:
                        c_array = np.zeros(6, dtype=np.float64)
                        mujoco.mj_contactForce(
                            m=state.mj_model,
                            d=state.mj_data,
                            id=contact_id,
                            result=c_array,
                        )

                        # Convert the contact force from contact frame to world frame
                        ref = np.reshape(contact.frame, (3, 3))
                        c_force = np.dot(np.linalg.inv(ref), c_array[0:3])

                        index = segment_capsule_geom_ids_to_contact_idx[contact.geom2]
                        ground_contacts[index] = max(
                            np.linalg.norm(c_force), ground_contacts[index]
                        )

            ground_contacts = (ground_contacts > 0).astype(int)
            return ground_contacts

        touch_observable = MJCObservable(
            name="segment_ground_contact",
            low=np.zeros(len(segment_capsule_geom_ids_to_contact_idx)),
            high=np.ones(len(segment_capsule_geom_ids_to_contact_idx)),
            retriever=get_segment_ground_contacts,
        )

        # torso framequat
        framequat_sensor = [
            sensor
            for sensor in sensors
            if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMEQUAT
        ][0]
        torso_rotation_observable = MJCObservable(
            name="torso_rotation",
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

        # torso framelinvel
        framelinvel_sensor = [
            sensor
            for sensor in sensors
            if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMELINVEL
        ][0]
        torso_linvel_observable = MJCObservable(
            name="torso_linear_velocity",
            low=-np.inf * np.ones(3),
            high=np.inf * np.ones(3),
            retriever=lambda state: np.array(
                state.mj_data.sensordata[
                    framelinvel_sensor.adr[0] : framelinvel_sensor.adr[0]
                    + framelinvel_sensor.dim[0]
                ]
            ),
        )

        # torso frameangvel
        frameangvel_sensor = [
            sensor
            for sensor in sensors
            if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMEANGVEL
        ][0]
        torso_angvel_observable = MJCObservable(
            name="torso_angular_velocity",
            low=-np.inf * np.ones(3),
            high=np.inf * np.ones(3),
            retriever=lambda state: np.array(
                state.mj_data.sensordata[
                    frameangvel_sensor.adr[0] : frameangvel_sensor.adr[0]
                    + frameangvel_sensor.dim[0]
                ]
            ),
        )

        # direction to target
        unit_xy_direction_to_target_observable = MJCObservable(
            name="unit_xy_direction_to_target",
            low=-np.ones(2),
            high=np.ones(2),
            retriever=lambda state: self._get_xy_direction_to_target(state=state)
            / self._get_xy_distance_to_target(state=state),
        )

        # distance to target
        xy_distance_to_target_observable = MJCObservable(
            name="xy_distance_to_target",
            low=np.zeros(1),
            high=np.inf * np.ones(1),
            retriever=lambda state: np.array(
                [self._get_xy_distance_to_target(state=state)]
            ),
        )

        return [
            in_plane_joint_pos_observable,
            out_of_plane_joint_pos_observable,
            in_plane_joint_velocity_observable,
            out_of_plane_joint_velocity_observable,
            touch_observable,
            torso_rotation_observable,
            torso_linvel_observable,
            torso_angvel_observable,
            unit_xy_direction_to_target_observable,
            xy_distance_to_target_observable,
        ]

    def _update_info(self, state: MJCEnvState) -> MJCEnvState:
        info = {"time": state.mj_data.time}

        return state.replace(info=info)

    def _update_reward(
        self, state: MJCEnvState, previous_state: MJCEnvState
    ) -> MJCEnvState:
        current_distance_to_target = self._get_xy_distance_to_target(state=state)
        previous_distance_to_target = self._get_xy_distance_to_target(
            state=previous_state
        )
        reward = previous_distance_to_target - current_distance_to_target

        return state.replace(reward=reward)

    def _update_terminated(self, state: MJCEnvState) -> bool:
        terminated = self._get_xy_distance_to_target(state=state) < 0.2

        return state.replace(terminated=terminated)

    def _update_truncated(self, state: MJCEnvState) -> bool:
        truncated = state.mj_data.time > self.environment_configuration.simulation_time

        return state.replace(truncated=truncated)

    def reset(self, rng: np.random.RandomState, *args, **kwargs) -> MJCEnvState:
        mj_model, mj_data = self._prepare_reset()

        # Set random target position
        angle = rng.uniform(0, 2 * np.pi)
        radius = self.environment_configuration.target_distance

        mj_model.body("target").pos = [
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.05,
        ]

        # Set morphology position
        mj_model.body("ToyExampleMorphology/torso").pos[2] = 0.11

        # Add noise to initial qpos and qvel of segment joints
        segment_joints = [
            mj_model.joint(joint_id)
            for joint_id in range(mj_model.njnt)
            if "segment" in mj_model.joint(joint_id).name
        ]
        num_segment_joints = len(segment_joints)
        joint_qpos_adrs = [joint.qposadr[0] for joint in segment_joints]
        mj_data.qpos[joint_qpos_adrs] = mj_model.qpos0[joint_qpos_adrs] + rng.uniform(
            low=-self.environment_configuration.joint_randomization_noise_scale,
            high=self.environment_configuration.joint_randomization_noise_scale,
            size=num_segment_joints,
        )
        joint_qvel_adrs = [joint.dofadr[0] for joint in segment_joints]
        mj_data.qvel[joint_qvel_adrs] = rng.uniform(
            low=-self.environment_configuration.joint_randomization_noise_scale,
            high=self.environment_configuration.joint_randomization_noise_scale,
            size=num_segment_joints,
        )

        state = self._finish_reset(models_and_datas=(mj_model, mj_data), rng=rng)
        return state
