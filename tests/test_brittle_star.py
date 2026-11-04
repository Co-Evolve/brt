"""Tests of the brittle star morphology, arena and tasks on the mjc, mjx and warp backends."""

import mujoco
import numpy as np
import pytest

import biorobot
from biorobot.brittle_star import (
    AquariumArena,
    AquariumArenaConfiguration,
    BrittleStarEnvConfig,
    BrittleStarMorphology,
    DirectedLocomotionEnvConfig,
    default_brittle_star_morphology_specification,
)
from biorobot.core import compose

TASKS = (
    "brittle_star/undirected_locomotion",
    "brittle_star/directed_locomotion",
    "brittle_star/light_escape",
)
#: The task whose MJX rollout is compared with MuJoCo C on every run. The tasks share the morphology and its physics,
#: and compiling MJX takes ~12 s per task, so the other tasks' comparisons only run with the slow tests. This one has
#: the most task code (the light map).
MJX_TASK = "brittle_star/light_escape"


def small_specification(**kwargs):
    defaults = dict(
        num_arms=5,
        num_segments_per_arm=2,
        use_p_control=True,
        radius_to_strength_factor=200,
    )
    return default_brittle_star_morphology_specification(**{**defaults, **kwargs})


def make(task, backend="mjc", **overrides):
    kwargs = dict(
        morphology_specification=small_specification(),
        simulation_time=0.4,
    )
    if task.endswith("light_escape"):
        kwargs["light_perlin_noise_scale"] = 10
    return biorobot.make(task, backend=backend, **{**kwargs, **overrides})


def test_registered():
    assert set(TASKS) <= set(biorobot.available())


def test_specification_validation():
    with pytest.raises(ValueError):
        default_brittle_star_morphology_specification(
            use_p_control=True, use_torque_control=True
        )
    with pytest.raises(ValueError):
        default_brittle_star_morphology_specification(
            use_p_control=True, use_tendons=True
        )
    with pytest.raises(ValueError):
        default_brittle_star_morphology_specification(
            num_arms=2, num_segments_per_arm=[1, 2, 3], use_p_control=True
        )


@pytest.mark.parametrize(
    "kwargs, num_actuators_per_segment",
    [
        (dict(use_p_control=True), 2),
        (dict(use_torque_control=True), 2),
        (
            dict(
                use_torque_control=True,
                use_tendons=True,
                num_contact_sensors_per_segment=8,
            ),
            4,
        ),
    ],
)
def test_morphology(kwargs, num_actuators_per_segment):
    specification = default_brittle_star_morphology_specification(
        num_arms=4, num_segments_per_arm=[3, 0, 2, 1], **kwargs
    )
    model = compose(AquariumArena(), BrittleStarMorphology(specification)).compile()
    num_segments = 6
    assert model.nu == num_actuators_per_segment * num_segments
    assert all(
        model.actuator(i).name.startswith("BrittleStarMorphology/")
        for i in range(model.nu)
    )
    num_touch = sum(model.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH)
    assert (
        num_touch
        == specification.sensor_specification.num_contact_sensors_per_segment
        * num_segments
    )
    assert (
        model.joint("BrittleStarMorphology/freejoint").type
        == mujoco.mjtJoint.mjJNT_FREE
    )
    if specification.actuation_specification.use_tendons:
        assert model.ntendon == 4 * num_segments
        # The first segment's tendons start at the disk taps of their own arm (also with empty arms in between).
        tendon = model.tendon("BrittleStarMorphology/arm_2_segment_0_tendon_0")
        first_site = model.wrap_objid[model.tendon_adr[tendon.id]]
        assert (
            model.site(first_site).name
            == "BrittleStarMorphology/central_disk_arm_2_tap_0"
        )


@pytest.mark.parametrize("task", TASKS)
def test_mjc_episode(task):
    env = make(task)
    kwargs = {"target_position": [0.5, 0.5, 0.05]} if "directed" in task else {}
    state = env.reset(0, **kwargs)
    rng = np.random.default_rng(0)
    rewards = []
    for _ in range(env.config.total_num_control_steps + 1):
        state = env.step(state, rng.uniform(env.action_low, env.action_high))
        rewards.append(float(state.reward))
        assert set(state.observations) == set(env.observation_space.spaces)
        for name, space in env.observation_space.spaces.items():
            assert state.observations[name].shape == space.shape
            assert np.all(np.isfinite(state.observations[name]))
        if state.truncated:
            break
    assert state.truncated and not state.terminated
    assert np.any(np.abs(rewards) > 1e-6)
    assert np.isclose(state.info["time"], state.data.time)


def test_factory_copies_config():
    """Overrides apply to a copy: the caller's config and later environments built from it are unaffected."""
    config = DirectedLocomotionEnvConfig(target_distance=3.0)
    env = make("brittle_star/directed_locomotion", config=config, target_distance=1.0)
    assert env.config.target_distance == 1.0 and config.target_distance == 3.0
    assert env.config is not config
    with pytest.raises(TypeError):
        make("brittle_star/directed_locomotion", config=BrittleStarEnvConfig())


def test_directed_target_and_termination():
    env = make("brittle_star/directed_locomotion")
    state = env.reset(0, target_position=[1.0, -2.0, 0.05])
    np.testing.assert_allclose(state.info["xy_target_position"], [1.0, -2.0])
    np.testing.assert_allclose(
        state.observations["xy_distance_to_target"], [np.sqrt(5)], atol=1e-6
    )
    # NaNs (or no target position) place the target at config.target_distance from the origin.
    state = env.reset(0, target_position=[np.nan] * 3)
    assert np.isclose(
        np.linalg.norm(state.info["xy_target_position"]), env.config.target_distance
    )
    # A target under the disk terminates the episode.
    state = env.reset(0, target_position=[0.0, 0.0, 0.05])
    state = env.step(state, np.zeros(env.mj_model.nu))
    assert state.terminated


def test_light_escape_light_map():
    env = make("brittle_star/light_escape", light_perlin_noise_scale=0)
    state = env.reset(0)
    light_map = state.info["_light_map"]
    assert light_map.shape == (150, 250)
    # Without noise, the light decreases linearly from 1 (x = -5) to 0 (x = 5) and the start is at x = -3.75.
    np.testing.assert_allclose(
        state.observations["disk_light_intake"], [0.875], atol=0.01
    )
    np.testing.assert_allclose(state.info["x_distance_travelled"], 0.0, atol=1e-6)
    with pytest.raises(ValueError):
        make("brittle_star/light_escape", light_perlin_noise_scale=7)
    with pytest.raises(ValueError):
        biorobot.make(
            "brittle_star/light_escape",
            arena_configuration=AquariumArenaConfiguration(sand_ground_color=False),
        )


@pytest.mark.parametrize(
    "task",
    [
        pytest.param(task, marks=() if task == MJX_TASK else pytest.mark.slow)
        for task in TASKS
    ],
)
def test_mjx_matches_mjc(task):
    jax = pytest.importorskip("jax")
    # A converged solver, so that differences between the MuJoCo C and MJX solvers do not dominate.
    overrides = dict(solver_iterations=50, solver_ls_iterations=50)
    env_mjc, env_mjx = make(task, **overrides), make(task, backend="mjx", **overrides)
    s_mjx = jax.jit(env_mjx.reset)(jax.random.PRNGKey(0))
    s_mjc = env_mjc.reset(0)
    # Start both from the MJX initial state (target, light map).
    s_mjc.data.qpos[:] = np.asarray(s_mjx.data.qpos)
    s_mjc.data.qvel[:] = np.asarray(s_mjx.data.qvel)
    s_mjc.data.mocap_pos[:] = np.asarray(s_mjx.data.mocap_pos)
    mujoco.mj_forward(env_mjc.mj_model, s_mjc.data)
    s_mjc.info.update(
        {k: np.asarray(v) for k, v in s_mjx.info.items() if k == "_light_map"}
    )

    step = jax.jit(env_mjx.step)
    rng = np.random.default_rng(0)
    for _ in range(10):
        action = rng.uniform(env_mjc.action_low, env_mjc.action_high)
        s_mjc = env_mjc.step(s_mjc, action)
        s_mjx = step(s_mjx, jax.numpy.asarray(action))
    for name, value in s_mjc.observations.items():
        np.testing.assert_allclose(
            np.asarray(s_mjx.observations[name]),
            value,
            atol=1e-3,
            rtol=1e-2,
            err_msg=name,
        )
    np.testing.assert_allclose(float(s_mjx.reward), float(s_mjc.reward), atol=1e-5)

    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    batch = jax.jit(jax.vmap(env_mjx.reset))(keys)
    batch = jax.jit(jax.vmap(env_mjx.step))(
        batch, jax.numpy.zeros((3, env_mjx.mj_model.nu))
    )
    for name, space in env_mjx.observation_space.spaces.items():
        assert batch.observations[name].shape == (3, *space.shape)
    assert batch.reward.shape == (3,)
    if task.endswith("light_escape"):
        # Every episode gets its own Perlin noise light map.
        assert not np.allclose(batch.info["_light_map"][0], batch.info["_light_map"][1])


@pytest.mark.parametrize("task", TASKS)
def test_render_rgb_array(task):
    env = make(
        task,
        render_mode="rgb_array",
        render_size=(48, 64),
        camera_ids=(0, 1),
        color_contacts=True,
    )
    state = env.reset(0)
    for _ in range(5):
        state = env.step(state, np.zeros(env.mj_model.nu))
    frames = env.render(state)
    assert len(frames) == 2 and frames[0].shape == (48, 64, 3)
    if task.endswith("light_escape"):
        model = env._render_model
        texture = model.texture("groundplane")
        texels = model.tex_data[texture.adr[0] : texture.adr[0] + 3 * texture.width[0]]
        # The bright (left) side of the first row is brighter than the dark (right) side.
        assert texels[:3].sum() > texels[-3:].sum()
    env.close()


def test_warp_matches_mjc():
    jax = pytest.importorskip("jax")
    # MJX ships its own copy of MuJoCo Warp; biorobot[warp] only adds NVIDIA Warp.
    pytest.importorskip("warp")
    task = "brittle_star/directed_locomotion"
    overrides = dict(solver_iterations=50, solver_ls_iterations=50)
    env_mjc, env_warp = make(task, **overrides), make(task, backend="warp", **overrides)
    s_warp = jax.jit(env_warp.reset)(jax.random.PRNGKey(0))
    s_mjc = env_mjc.reset(0)
    s_mjc.data.qpos[:] = np.asarray(s_warp.data.qpos)
    s_mjc.data.mocap_pos[:] = np.asarray(s_warp.data.mocap_pos)
    mujoco.mj_forward(env_mjc.mj_model, s_mjc.data)
    step = jax.jit(env_warp.step)
    for _ in range(5):
        s_mjc = env_mjc.step(s_mjc, np.zeros(env_mjc.mj_model.nu))
        s_warp = step(s_warp, jax.numpy.zeros(env_warp.mj_model.nu))
    for name in ("joint_position", "disk_position", "xy_distance_to_target"):
        np.testing.assert_allclose(
            np.asarray(s_warp.observations[name]), s_mjc.observations[name], atol=1e-3
        )


@pytest.mark.slow
def test_warp_batched_matches_mjx():
    """Vmapped warp environments keep all their contacts (MuJoCo Warp shares one contact buffer between worlds)."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("warp")
    num_envs = 8
    overrides = dict(solver_iterations=50, solver_ls_iterations=50, num_envs=num_envs)
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    disk_positions = []
    for backend in ("mjx", "warp"):
        env = make("brittle_star/undirected_locomotion", backend=backend, **overrides)
        batch = jax.jit(jax.vmap(env.reset))(keys)
        step = jax.jit(jax.vmap(env.step))
        for _ in range(20):
            batch = step(batch, jax.numpy.zeros((num_envs, env.mj_model.nu)))
        disk_positions.append(np.asarray(batch.observations["disk_position"]))
    np.testing.assert_allclose(disk_positions[1], disk_positions[0], atol=1e-3)
