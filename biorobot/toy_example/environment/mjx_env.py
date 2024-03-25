from __future__ import annotations

from typing import Any, Dict, List

import chex
import jax.random
from jax import numpy as jnp
from jax._src.scipy.spatial.transform import Rotation
from mujoco._structs import _MjModelJointViews
from moojoco.environment.mjx_env import MJXEnv, MJXEnvState, MJXObservable

from biorobot.toy_example.environment.mjc_env import ToyExampleEnvironmentConfiguration
from biorobot.toy_example.mjcf.arena.arena import MJCFPlaneWithTargetArena
from biorobot.toy_example.mjcf.morphology.morphology import MJCFToyExampleMorphology


class ToyExampleMJXEnvironment(MJXEnv):
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
        self._segment_joints: List[_MjModelJointViews] = [
            self.frozen_mj_model.joint(joint_id)
            for joint_id in range(self.frozen_mj_model.njnt)
            if "segment" in self.frozen_mj_model.joint(joint_id).name
        ]
        self._segment_joint_qpos_adrs = jnp.array(
            [joint.qposadr[0] for joint in self._segment_joints]
        )
        self._segment_joint_qvel_adrs = jnp.array(
            [joint.dofadr[0] for joint in self._segment_joints]
        )

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: MJCFToyExampleMorphology,
        arena: MJCFPlaneWithTargetArena,
        configuration: ToyExampleEnvironmentConfiguration,
    ) -> ToyExampleMJXEnvironment:
        return super().from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=configuration
        )

    @property
    def environment_configuration(self) -> ToyExampleEnvironmentConfiguration:
        return super().environment_configuration

    @staticmethod
    def _get_xy_direction_to_target(
        state: MJXEnvState,
    ) -> jnp.ndarray:
        torso_body_id = state.mj_model.body("ToyExampleMorphology/torso").id
        target_body_id = state.mj_model.body("target").id

        torso_body_pos = state.mjx_data.xpos[torso_body_id]
        target_pos = state.mjx_data.xpos[target_body_id]

        return (target_pos - torso_body_pos)[:2]

    @staticmethod
    def _get_xy_distance_to_target(state: MJXEnvState) -> float:
        xy_direction_to_target = ToyExampleMJXEnvironment._get_xy_direction_to_target(
            state=state
        )
        return jnp.linalg.norm(xy_direction_to_target)

    def _create_observables(self) -> List[MJXObservable]:
        in_plane_joints = [
            self.frozen_mj_model.joint(joint_id)
            for joint_id in range(self.frozen_mj_model.njnt)
            if "in_plane_joint" in self.frozen_mj_model.joint(joint_id).name
        ]
        out_of_plane_joints = [
            self.frozen_mj_model.joint(joint_id)
            for joint_id in range(self.frozen_mj_model.njnt)
            if "out_of_plane_joint" in self.frozen_mj_model.joint(joint_id).name
        ]

        in_plane_joint_qpos_adr = jnp.array(
            [joint.qposadr[0] for joint in in_plane_joints]
        )
        in_plane_joint_range = jnp.array([joint.range for joint in in_plane_joints]).T
        out_of_plane_joint_qpos_adr = jnp.array(
            [joint.qposadr[0] for joint in out_of_plane_joints]
        )
        out_of_plane_joint_range = jnp.array(
            [joint.range for joint in out_of_plane_joints]
        ).T
        in_plane_joint_qvel_adr = jnp.array(
            [joint.dofadr[0] for joint in in_plane_joints]
        )
        out_of_plane_joint_qvel_adr = jnp.array(
            [joint.dofadr[0] for joint in out_of_plane_joints]
        )

        in_plane_joint_position_observable = MJXObservable(
            name="in_plane_joint_position",
            low=in_plane_joint_range[0],
            high=in_plane_joint_range[1],
            retriever=lambda state: state.mjx_data.qpos[in_plane_joint_qpos_adr],
        )

        out_of_plane_joint_position_observable = MJXObservable(
            name="out_of_plane_joint_position",
            low=out_of_plane_joint_range[0],
            high=out_of_plane_joint_range[1],
            retriever=lambda state: state.mjx_data.qpos[out_of_plane_joint_qpos_adr],
        )

        in_plane_joint_velocity_observable = MJXObservable(
            name="in_plane_joint_velocity",
            low=-jnp.inf * jnp.ones(len(in_plane_joints)),
            high=jnp.inf * jnp.ones(len(in_plane_joints)),
            retriever=lambda state: state.mjx_data.qvel[in_plane_joint_qvel_adr],
        )
        out_of_plane_joint_velocity_observable = MJXObservable(
            name="out_of_plane_joint_velocity",
            low=-jnp.inf * jnp.ones(len(out_of_plane_joints)),
            high=jnp.inf * jnp.ones(len(out_of_plane_joints)),
            retriever=lambda state: state.mjx_data.qvel[out_of_plane_joint_qvel_adr],
        )

        segment_capsule_geom_ids = jnp.array(
            [
                geom_id
                for geom_id in range(self.frozen_mj_model.ngeom)
                if "segment" in self.frozen_mj_model.geom(geom_id).name
                and "capsule" in self.frozen_mj_model.geom(geom_id).name
            ]
        )
        ground_floor_geom_id = self.frozen_mj_model.geom("groundplane").id

        def get_segment_ground_contacts(state: MJXEnvState) -> jnp.ndarray:
            contact_data = state.mjx_data.contact
            contacts = contact_data.dist <= 0
            valid_geom1s = contact_data.geom1 == ground_floor_geom_id

            def solve_contact(geom_id: int) -> jnp.ndarray:
                return (
                    jnp.sum(contacts * valid_geom1s * (contact_data.geom2 == geom_id))
                    > 0
                ).astype(int)

            return jax.vmap(solve_contact)(segment_capsule_geom_ids)

        touch_observable = MJXObservable(
            name="segment_ground_contact",
            low=jnp.zeros(len(segment_capsule_geom_ids)),
            high=jnp.ones(len(segment_capsule_geom_ids)),
            retriever=get_segment_ground_contacts,
        )

        # torso framequat
        torso_id = self.frozen_mj_model.body("ToyExampleMorphology/torso").id
        torso_rotation_observable = MJXObservable(
            name="torso_rotation",
            low=-jnp.pi * jnp.ones(3),
            high=jnp.pi * jnp.ones(3),
            retriever=lambda state: Rotation.from_quat(
                quat=state.mjx_data.xquat[torso_id]
            ).as_euler(seq="xyz"),
        )

        # torso framelinvel
        morphology_freejoint_adr = self.frozen_mj_model.joint(
            "ToyExampleMorphology/freejoint/"
        ).dofadr[0]
        torso_linvel_observable = MJXObservable(
            name="torso_linear_velocity",
            low=-jnp.inf * jnp.ones(3),
            high=jnp.inf * jnp.ones(3),
            retriever=lambda state: state.mjx_data.qvel[
                morphology_freejoint_adr : morphology_freejoint_adr + 3
            ],
        )
        # torso frameangvel
        torso_angvel_observable = MJXObservable(
            name="torso_angular_velocity",
            low=-jnp.inf * jnp.ones(3),
            high=jnp.inf * jnp.ones(3),
            retriever=lambda state: state.mjx_data.qvel[
                morphology_freejoint_adr + 3 : morphology_freejoint_adr + 6
            ],
        )

        # direction to target
        unit_xy_direction_to_target_observable = MJXObservable(
            name="unit_xy_direction_to_target",
            low=-jnp.ones(2),
            high=jnp.ones(2),
            retriever=lambda state: self._get_xy_direction_to_target(state=state)
            / self._get_xy_distance_to_target(state=state),
        )
        # distance to target
        xy_distance_to_target_observable = MJXObservable(
            name="xy_distance_to_target",
            low=jnp.zeros(1),
            high=jnp.inf * jnp.ones(1),
            retriever=lambda state: self._get_xy_distance_to_target(state=state),
        )

        return [
            in_plane_joint_position_observable,
            out_of_plane_joint_position_observable,
            in_plane_joint_velocity_observable,
            out_of_plane_joint_velocity_observable,
            touch_observable,
            torso_rotation_observable,
            torso_linvel_observable,
            torso_angvel_observable,
            unit_xy_direction_to_target_observable,
            xy_distance_to_target_observable,
        ]

    def _update_info(self, state: MJXEnvState) -> MJXEnvState:
        info = {"time": state.mjx_data.time}

        return state.replace(info=info)

    def _update_reward(
        self, state: MJXEnvState, previous_state: MJXEnvState
    ) -> MJXEnvState:
        current_distance_to_target = self._get_xy_distance_to_target(state=state)
        previous_distance_to_target = self._get_xy_distance_to_target(
            state=previous_state
        )
        reward = previous_distance_to_target - current_distance_to_target

        return state.replace(reward=reward)

    def _update_terminated(self, state: MJXEnvState) -> bool:
        terminated = self._get_xy_distance_to_target(state=state) < 0.2

        return state.replace(terminated=terminated)

    def _update_truncated(self, state: MJXEnvState) -> bool:
        truncated = state.mjx_data.time > self.environment_configuration.simulation_time

        return state.replace(truncated=truncated)

    def _get_random_target_position(self, rng: jnp.ndarray) -> jnp.ndarray:
        angle = jax.random.uniform(key=rng, shape=(), minval=0, maxval=jnp.pi * 2)
        radius = self.environment_configuration.target_distance
        target_pos = jnp.array([radius * jnp.cos(angle), radius * jnp.sin(angle), 0.05])
        return target_pos

    def reset(self, rng: chex.PRNGKey, *args, **kwargs) -> MJXEnvState:
        rng, target_pos_rng, qpos_rng, qvel_rng = jax.random.split(key=rng, num=4)

        (mj_model, mj_data), (mjx_model, mjx_data) = self._prepare_reset()

        target_body_id = mj_model.body("target").id
        torso_body_id = mj_model.body("ToyExampleMorphology/torso").id

        # Set random target position
        target_pos = self._get_random_target_position(rng=target_pos_rng)
        mjx_model = mjx_model.replace(
            body_pos=mjx_model.body_pos.at[target_body_id].set(target_pos)
        )

        # Set morphology position
        morphology_pos = jnp.array([0.0, 0.0, 0.11])
        mjx_model = mjx_model.replace(
            body_pos=mjx_model.body_pos.at[torso_body_id].set(morphology_pos)
        )

        # Add noise to initial qpos and qvel of segment joints
        qpos = jnp.copy(mjx_model.qpos0)
        qvel = jnp.zeros(mjx_model.nv)

        qpos = qpos.at[self._segment_joint_qpos_adrs].set(
            qpos[self._segment_joint_qpos_adrs]
            + jax.random.uniform(
                key=qpos_rng,
                shape=(len(self._segment_joints),),
                minval=-self.environment_configuration.joint_randomization_noise_scale,
                maxval=self.environment_configuration.joint_randomization_noise_scale,
            )
        )
        qvel = qvel.at[self._segment_joint_qvel_adrs].set(
            jax.random.uniform(
                key=qvel_rng,
                shape=(len(self._segment_joints),),
                minval=-self.environment_configuration.joint_randomization_noise_scale,
                maxval=self.environment_configuration.joint_randomization_noise_scale,
            )
        )
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(mjx_model.nu))

        state = self._finish_reset(
            models_and_datas=((mj_model, mj_data), (mjx_model, mjx_data)), rng=rng
        )
        return state
