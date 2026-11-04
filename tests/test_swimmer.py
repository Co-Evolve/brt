"""Tests of the swimmer: its morphology, its head-first swimming task and its backends."""

import mujoco
import numpy as np
import pytest

import biorobot
from biorobot.swimmer import (
    SwimmerMorphology,
    default_swimmer_morphology_specification,
)

TASK = "swimmer/head_first_swimming"
PREFIX = "SwimmerMorphology/"


def travelling_wave(
    num_joints: int, t: float, head_to_tail: bool = True, amplitude: float = 0.6
):
    """Servo targets (rad) of a body wave with one wavelength along the body, at 1.5 Hz."""
    phase = 2 * np.pi * np.arange(num_joints) / num_joints
    return amplitude * np.sin(2 * np.pi * 1.5 * t - (phase if head_to_tail else -phase))


def run_episode(env, head_to_tail=True):
    state = env.reset(0)
    total_reward = 0.0
    while not state.truncated:
        action = travelling_wave(
            env.mj_model.nu, float(state.info["time"]), head_to_tail
        )
        state = env.step(state, action)
        total_reward += float(state.reward)
    return state, total_reward


# ----------------------------------------------------------------------------------------------------------------------
# Morphology
# ----------------------------------------------------------------------------------------------------------------------
def test_registry():
    assert TASK in biorobot.available()


@pytest.mark.parametrize("num_segments", [4, 12])
def test_morphology_counts(num_segments):
    model = SwimmerMorphology(
        default_swimmer_morphology_specification(num_segments=num_segments)
    ).compile()
    num_joints = num_segments - 1
    assert model.nbody == 1 + num_segments
    assert (
        model.njnt == num_joints
    )  # the free joint is added when the swimmer is attached to an arena
    assert model.nu == num_joints
    # Joint position, velocity and actuator force per joint, and 4 frame sensors on the head.
    assert model.nsensor == 3 * num_joints + 4
    assert np.all(model.geom_fluid[:, 0] == 1)


def test_tapered_segments():
    specification = default_swimmer_morphology_specification(num_segments=5, taper=0.5)
    np.testing.assert_allclose(
        [s.length for s in specification.segment_specifications],
        [0.1, 0.0875, 0.075, 0.0625, 0.05],
    )
    assert specification.body_length == pytest.approx(0.375)


def test_specification_validation():
    specification = default_swimmer_morphology_specification(num_segments=4)
    with pytest.raises(ValueError):
        type(specification)(
            specification.segment_specifications,
            specification.joint_specifications[:-1],
        )
    with pytest.raises(ValueError):
        default_swimmer_morphology_specification(fluid_model="potential_flow")
    inertia = default_swimmer_morphology_specification(fluid_model="inertia")
    assert np.all(SwimmerMorphology(inertia).compile().geom_fluid[:, 0] == 0)


# ----------------------------------------------------------------------------------------------------------------------
# MJC
# ----------------------------------------------------------------------------------------------------------------------
def test_mjc_episode():
    env = biorobot.make(TASK, simulation_time=2.0)
    assert env.config.total_num_control_steps == 1000
    state, total_reward = run_episode(env)
    assert not state.terminated
    assert state.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0
    assert set(state.observations) == set(env.observation_space.spaces)
    for name, space in env.observation_space.spaces.items():
        observation = np.asarray(state.observations[name], dtype=np.float64)
        assert observation.shape == space.shape, name
        assert space.contains(observation), name
    assert env.observation_space["joint_position"].shape == (7,)
    # The rewards add up to the head's displacement along -x.
    assert total_reward == pytest.approx(float(state.info["head_first_distance"]))
    assert float(state.info["head_first_distance"]) == pytest.approx(
        -state.observations["head_position"][0]
    )
    # The swimmer stays in the horizontal plane.
    assert abs(state.observations["head_position"][2]) < 1e-9


def test_wave_direction_sets_swimming_direction():
    """A head-to-tail body wave drives the swimmer head first, a tail-to-head wave tail first."""
    env = biorobot.make(TASK, simulation_time=3.0)
    head_first, _ = run_episode(env, head_to_tail=True)
    head_first_distance = float(head_first.info["head_first_distance"])
    tail_first, _ = run_episode(env, head_to_tail=False)
    assert head_first_distance > 0.05
    assert float(tail_first.info["head_first_distance"]) < -0.05


def test_actuator_work():
    """Actuator work is the time integral of sum |torque * joint velocity|, and 0 without actuation."""
    env = biorobot.make(TASK, simulation_time=0.5)
    state = env.reset(0)
    work = 0.0
    while not state.truncated:
        state = env.step(state, travelling_wave(7, float(state.info["time"])))
        dofs = env._joint_dof_adrs
        work += np.sum(
            np.abs(state.data.qfrc_actuator[dofs] * state.data.qvel[dofs])
        ) * (env.config.control_timestep)
    assert float(state.info["actuator_work"]) == pytest.approx(work)
    assert work > 0

    state = env.reset(0)
    for _ in range(50):
        state = env.step(state, np.zeros(env.mj_model.nu))
    assert float(state.info["actuator_work"]) == 0.0


def test_render_rgb_array():
    env = biorobot.make(TASK, render_mode="rgb_array", render_size=(48, 64))
    state = env.reset(0)
    frames = env.render(state)
    assert len(frames) == 1 and frames[0].shape == (48, 64, 3)
    env.close()


# ----------------------------------------------------------------------------------------------------------------------
# MJX and MuJoCo Warp
# ----------------------------------------------------------------------------------------------------------------------
def test_mjx_rejects_ellipsoid_fluid_model():
    pytest.importorskip("jax")
    with pytest.raises(NotImplementedError, match="ellipsoid fluid model"):
        biorobot.make(TASK, backend="mjx")


def _compare_with_mjc(backend, num_steps=200, **make_kwargs):
    jax = pytest.importorskip("jax")
    env_mjc = biorobot.make(TASK, **make_kwargs)
    env_jax = biorobot.make(TASK, backend=backend, **make_kwargs)
    s_mjc, s_jax = env_mjc.reset(0), jax.jit(env_jax.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env_jax.step)
    for i in range(num_steps):
        action = travelling_wave(
            env_mjc.mj_model.nu, i * env_mjc.config.control_timestep
        )
        s_mjc = env_mjc.step(s_mjc, action)
        s_jax = step(s_jax, jax.numpy.asarray(action, dtype=jax.numpy.float32))
    for name in ("joint_position", "head_position"):
        np.testing.assert_allclose(
            np.asarray(s_jax.observations[name]), s_mjc.observations[name], atol=1e-3
        )
    for name in ("head_first_distance", "actuator_work"):
        assert float(s_jax.info[name]) == pytest.approx(
            float(s_mjc.info[name]), rel=1e-2, abs=1e-4
        )
    return s_jax


def test_mjx_with_inertia_fluid_model_matches_mjc():
    """With the inertia-based fluid model, which MJX supports, MJX matches MuJoCo C."""
    jax = pytest.importorskip("jax")
    specification = default_swimmer_morphology_specification(fluid_model="inertia")
    _compare_with_mjc("mjx", morphology_specification=specification)

    env = biorobot.make(TASK, backend="mjx", morphology_specification=specification)
    batch = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(1), 4))
    batch = jax.jit(jax.vmap(env.step))(batch, jax.numpy.zeros((4, env.mj_model.nu)))
    assert batch.info["actuator_work"].shape == (4,)


def test_warp_matches_mjc():
    """MuJoCo Warp supports the ellipsoid fluid model and matches MuJoCo C."""
    pytest.importorskip("jax")
    # MJX ships its own copy of MuJoCo Warp; biorobot[warp] only adds NVIDIA Warp.
    pytest.importorskip("warp")
    _compare_with_mjc("warp")
