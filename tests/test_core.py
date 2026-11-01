"""Tests of the backend-agnostic core on a tiny toy environment."""

import mujoco
import numpy as np
import pytest

from biorobot.core import (
    Arena,
    BRTEnv,
    EnvConfig,
    Morphology,
    Observable,
    make_config,
    sensor_observable,
)
from biorobot.core.math import quat_to_euler


class ToyMorphology(Morphology):
    def __init__(self):
        super().__init__(specification=None, name="Toy")

    def _build(self):
        torso = self.spec.worldbody.add_body(name="torso")
        torso.add_geom(name="torso_geom", size=[0.1, 0, 0])
        arm = torso.add_body(name="arm", pos=[0.1, 0, 0])
        arm.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0.2, 0, 0],
            size=[0.02, 0, 0],
        )
        arm.add_joint(name="hinge", axis=[0, 1, 0], range=[-1, 1], armature=0.01)
        self.spec.add_actuator(
            name="motor",
            target="hinge",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrlrange=[-1, 1],
            ctrllimited=True,
        )
        self.spec.add_sensor(
            name="hinge_pos",
            type=mujoco.mjtSensor.mjSENS_JOINTPOS,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname="hinge",
        )


class ToyArena(Arena):
    def __init__(self):
        super().__init__(name="ToyArena")

    def _build(self):
        self.spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
        self.spec.worldbody.add_camera(
            name="cam", pos=[0, -3, 1], xyaxes=[1, 0, 0, 0, 0.3, 1]
        )


class ToyEnv(BRTEnv):
    morphology_name = "Toy"

    def _cache_references(self, mj_model):
        self._torso_id = mj_model.body("Toy/torso").id
        self._free_qpos = mj_model.joint("Toy/freejoint").qposadr[0]

    def _create_observables(self):
        return [
            sensor_observable(
                "hinge_position", self.mj_model, mujoco.mjtSensor.mjSENS_JOINTPOS
            ),
            Observable(
                "height",
                np.zeros(1),
                np.full(1, np.inf),
                lambda data, info: data.xpos[self._torso_id, 2],
            ),
        ]

    def _reset(self, data, rng):
        height, rng = self.backend.uniform(rng, 0.5, 1.0)
        qpos = self.backend.at_set(data.qpos, self._free_qpos + 2, height)
        data = self.backend.replace(data, qpos=qpos)
        return data, {"start_height": height}, rng

    def _get_reward(self, state, previous_state):
        return (
            previous_state.observations["height"][0] - state.observations["height"][0]
        )


def make_env(backend, simulation_time=0.5, **config_overrides):
    config = EnvConfig(
        simulation_time=simulation_time,
        num_physics_steps_per_control_step=5,
        **config_overrides,
    )
    return ToyEnv.from_morphology_and_arena(
        ToyMorphology(), ToyArena(), config, backend=backend
    )


def test_mjc_episode():
    env = make_env("mjc")
    state = env.reset(0)
    assert 0.5 <= state.info["start_height"] <= 1.0
    total = 0.0
    for _ in range(env.config.total_num_control_steps):
        state = env.step(state, np.ones(1))
        total += float(state.reward)
    assert state.truncated
    assert total > 0  # The toy falls, so the height decreases.
    assert set(state.observations) == set(env.observation_space.spaces)


@pytest.mark.parametrize("backend", ["mjx"])
def test_jax_backend_matches_mjc(backend):
    jax = pytest.importorskip("jax")
    env_mjc, env_jax = make_env("mjc"), make_env(backend)
    s_mjc = env_mjc.reset(0)
    s_jax = env_jax.reset(jax.random.PRNGKey(0))
    # Same initial height so that both backends simulate the same rollout.
    qpos = np.asarray(s_jax.data.qpos)
    s_mjc.data.qpos[:] = qpos
    mujoco.mj_forward(env_mjc.mj_model, s_mjc.data)
    step = jax.jit(env_jax.step)
    for _ in range(20):
        s_mjc = env_mjc.step(s_mjc, np.ones(1))
        s_jax = step(s_jax, jax.numpy.ones(1))
    np.testing.assert_allclose(
        s_mjc.observations["height"], s_jax.observations["height"], atol=1e-4
    )

    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    batch = jax.vmap(env_jax.reset)(keys)
    batch = jax.jit(jax.vmap(env_jax.step))(batch, jax.numpy.zeros((4, 1)))
    assert batch.observations["height"].shape == (4, 1)


@pytest.mark.parametrize("backend", ["mjc", "mjx"])
@pytest.mark.parametrize("simulation_time", [0.5, 2.0, 5.0])
def test_truncation_after_total_num_control_steps(backend, simulation_time):
    """Episodes last exactly total_num_control_steps steps, despite round-off in data.time (float64 or float32)."""
    env = make_env(backend, simulation_time=simulation_time, physics_timestep=0.002)
    if backend == "mjc":
        state, step = env.reset(0), env.step
    else:
        jax = pytest.importorskip("jax")
        state, step = env.reset(jax.random.PRNGKey(0)), jax.jit(env.step)
    num_steps = 0
    while not state.truncated:
        state = step(state, np.zeros(1))
        num_steps += 1
    assert num_steps == env.config.total_num_control_steps


def test_make_config():
    config = EnvConfig(simulation_time=3.0)
    copy = make_config(EnvConfig, config, simulation_time=1.0)
    assert copy.simulation_time == 1.0 and config.simulation_time == 3.0
    assert make_config(EnvConfig).simulation_time == EnvConfig().simulation_time
    with pytest.raises(TypeError):
        make_config(EnvConfig, config, control_timestep=0.1)  # a property, not a field
    with pytest.raises(TypeError):
        make_config(EnvConfig, config, unknown=1)

    class OtherConfig(EnvConfig):
        pass

    with pytest.raises(TypeError):
        make_config(OtherConfig, config)


def test_mjx_reset_accepts_numpy_seed():
    pytest.importorskip("jax")
    env = make_env("mjx")
    state = env.reset(np.int64(3))
    assert np.isfinite(state.observations["height"]).all()


def test_quat_to_euler():
    euler = np.array([0.3, -0.4, 1.2])
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, euler, "XYZ")  # extrinsic, like quat_to_euler
    np.testing.assert_allclose(quat_to_euler(quat, np), euler, atol=1e-8)


def test_render_rgb_array():
    env = make_env("mjc")
    env.config.render_mode = "rgb_array"
    env.config.render_size = (64, 80)
    frames = env.render(env.reset(0))
    assert len(frames) == 1 and frames[0].shape == (64, 80, 3)
    env.close()


def test_render_batched_state():
    """A batched state is rendered with one renderer; each frame matches rendering that environment on its own."""
    jax = pytest.importorskip("jax")
    env = make_env("mjx", render_mode="rgb_array", render_size=(32, 40))
    batch = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 3))
    frames = env.render(batch)
    assert len(frames) == 3
    renderer = env._renderer
    single = jax.tree_util.tree_map(lambda x: x[1], batch)
    np.testing.assert_array_equal(env.render(single)[0], frames[1])
    assert env._renderer is renderer
    env.close()
