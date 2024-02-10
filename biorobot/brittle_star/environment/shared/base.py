import jax.numpy as jnp
import mujoco


class BrittleStarEnvironmentBase:
    def __init__(self) -> None:
        self._segment_joints_qpos_adrs = None
        self._segment_joints_qvel_adrs = None
        self._segment_capsule_geom_ids = None

    def _cache_references(self, mj_model: mujoco.MjModel) -> None:
        self._get_segment_capsule_geom_ids(mj_model=mj_model)
        self._get_segment_joints_qpos_adrs(mj_model=mj_model)
        self._get_segment_joints_qvel_adrs(mj_model=mj_model)

    def _get_segment_joints_qpos_adrs(self, mj_model: mujoco.MjModel) -> jnp.ndarray:
        if self._segment_joints_qpos_adrs is None:
            self._segment_joints_qpos_adrs = jnp.array(
                [
                    mj_model.joint(joint_id).qposadr[0]
                    for joint_id in range(mj_model.njnt)
                    if "segment" in mj_model.joint(joint_id).name
                ]
            )
        return self._segment_joints_qpos_adrs

    def _get_segment_joints_qvel_adrs(self, mj_model: mujoco.MjModel) -> jnp.ndarray:
        if self._segment_joints_qvel_adrs is None:
            self._segment_joints_qvel_adrs = jnp.array(
                [
                    mj_model.joint(joint_id).dofadr[0]
                    for joint_id in range(mj_model.njnt)
                    if "segment" in mj_model.joint(joint_id).name
                ]
            )
        return self._segment_joints_qvel_adrs

    def _get_segment_capsule_geom_ids(self, mj_model: mujoco.MjModel) -> jnp.ndarray:
        if self._segment_capsule_geom_ids is None:
            self._segment_capsule_geom_ids = jnp.array(
                [
                    geom_id
                    for geom_id in range(mj_model.ngeom)
                    if "segment" in mj_model.geom(geom_id).name
                    and "capsule" in mj_model.geom(geom_id).name
                ]
            )
        return self._segment_capsule_geom_ids
