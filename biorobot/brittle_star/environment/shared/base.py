import abc
from typing import List, Union

import chex
import jax.numpy as jnp
import mujoco
from moojoco.environment.base import MuJoCoEnvironmentConfiguration, BaseEnvState
from moojoco.environment.renderer import MujocoRenderer

from biorobot.utils import colors


class BrittleStarEnvironmentBaseConfiguration(MuJoCoEnvironmentConfiguration):
    def __init__(
            self,
            joint_randomization_noise_scale: float = 0.0,
            hfield_perlin_noise_scale: int = 0,
            color_contacts: bool = False,
            solver_iterations: int = 1,
            solver_ls_iterations: int = 5,
            *args,
            **kwargs,
    ) -> None:
        assert hfield_perlin_noise_scale == 0 or 200 % hfield_perlin_noise_scale == 0, (
            "Please only provide integer factors of 200 for the "
            "'hfield_perlin_noise_scale' parameter."
        )
        super().__init__(
            disable_eulerdamp=True,
            solver_iterations=solver_iterations,
            solver_ls_iterations=solver_ls_iterations,
            *args,
            **kwargs,
        )
        self.joint_randomization_noise_scale = joint_randomization_noise_scale
        self.hfield_perlin_noise_scale = hfield_perlin_noise_scale
        self.color_contacts = color_contacts


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

    def _color_segment_capsule_contacts(
            self, mj_models: List[mujoco.MjModel], contact_bools: chex.Array
    ) -> None:
        for i, mj_model in enumerate(mj_models):
            if len(contact_bools.shape) > 1:
                contacts = contact_bools[i]
            else:
                contacts = contact_bools

            for capsule_geom_id, contact in zip(
                    self._segment_capsule_geom_ids, contacts
            ):
                if contact:
                    mj_model.geom(capsule_geom_id).rgba = colors.rgba_red
                else:
                    mj_model.geom(capsule_geom_id).rgba = colors.rgba_green

    def _update_renderer_context(
            self,
            mj_model: mujoco.MjModel,
            state: BaseEnvState,
            renderer: Union[MujocoRenderer, mujoco.Renderer],
    ) -> None:
        if jnp.any(state.info.get("_hfield_has_changed", False)):
            ground_hfield = mj_model.hfield("groundplane_hfield")
            context = self.get_renderer_context(renderer=renderer)
            mujoco.mjr_uploadHField(m=mj_model, con=context, hfieldid=ground_hfield.id)

    @abc.abstractmethod
    def _update_hfield(self, state: BaseEnvState) -> BaseEnvState:
        raise NotImplementedError
