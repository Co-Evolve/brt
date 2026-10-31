"""Tests of the jumping spider morphology and its long jump, platform jump and directed jump environments."""

import mujoco
import numpy as np
import pytest

import biorobot
from biorobot.jumping_spider import (
    JumpingSpiderMorphology,
    default_jumping_spider_morphology_specification,
)
from biorobot.jumping_spider.examples.scripted_jump import scripted_jump_action

TASKS = [
    "jumping_spider/long_jump",
    "jumping_spider/platform_jump",
    "jumping_spider/directed_jump",
]
#: Reset options that make the initial state deterministic, so that backends can be compared.
FIXED_RESET_OPTIONS = {
    "jumping_spider/long_jump": {},
    "jumping_spider/platform_jump": {"gap": 0.15},
    "jumping_spider/directed_jump": {"target_position": [0.3, 0.0, 0.15]},
}
BODY_LENGTH = 0.132
#: Angular velocity (rad/s) of the airborne body that a stable simulation never reaches (it stays below ~150 rad/s);
#: MuJoCo's divergence warning only fires once the state has already exploded.
MAX_BODY_ANGULAR_VELOCITY = 1000.0


def make(task, backend="mjc", **kwargs):
    if task == "jumping_spider/long_jump":
        # Random actions make the spider hop, which would end the episode at the first landing.
        kwargs.setdefault("terminate_on_landing", False)
    return biorobot.make(task, backend=backend, **kwargs)


def rollout(env, policy, reset_options=None, seed=0):
    """Run one mjc episode; returns the final state, the rewards and the maximum flight time."""
    state = env.reset(seed, **(reset_options or {}))
    rewards, max_flight_time = [], 0.0
    while not (state.terminated or state.truncated):
        state = env.step(state, policy(state))
        rewards.append(float(state.reward))
        max_flight_time = max(max_flight_time, float(state.info["flight_time"]))
        assert state.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0
    return state, np.array(rewards), max_flight_time


def random_policy(env, seed=0):
    rng = np.random.default_rng(seed)
    return lambda state: rng.uniform(env.action_low, env.action_high)


def bang_bang_policy(env, seed=0):
    """Every action independently at its lower or upper bound, as saturated (squashed or clipped) policies produce."""
    rng = np.random.default_rng(seed)
    return lambda state: np.where(
        rng.random(env.mj_model.nu) < 0.5, env.action_low, env.action_high
    )


def assert_stable(qvel):
    qvel = np.asarray(qvel)
    assert np.all(np.isfinite(qvel))
    assert np.all(np.abs(qvel[..., 3:6]) < MAX_BODY_ANGULAR_VELOCITY)


def scripted_policy(env):
    return lambda state: scripted_jump_action(
        float(state.info["time"]), env.actuator_names
    )


# ----------------------------------------------------------------------------------------------------------------------
# Morphology
# ----------------------------------------------------------------------------------------------------------------------
def test_morphology():
    model = JumpingSpiderMorphology(
        default_jumping_spider_morphology_specification()
    ).compile()
    # 8 legs x 12 joints + 2 abdomen joints, each with one actuator.
    assert model.njnt == model.nu == 98
    assert np.all(model.dof_armature > 0)
    assert np.all(model.dof_damping > 0)
    assert np.all(model.jnt_range[:, 0] < model.jnt_range[:, 1])

    mass = {model.body(i).name: model.body_mass[i] for i in range(model.nbody)}
    legs = sum(m for name, m in mass.items() if name.startswith("leg_"))
    total = model.body_subtreemass[1]
    assert 0.25 < total < 0.4
    assert 0.5 < mass["abdomen"] / mass["cephalothorax"] < 2.0
    assert legs / total < 0.3


def test_mirrored_joint_convention():
    """Equal joint angles on both sides of the body give mirrored poses."""
    model = JumpingSpiderMorphology(
        default_jumping_spider_morphology_specification()
    ).compile()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    for pair in range(1, 5):
        left = [j for j in range(model.njnt) if f"leg_L{pair}_" in model.joint(j).name]
        right = [j for j in range(model.njnt) if f"leg_R{pair}_" in model.joint(j).name]
        angles = rng.uniform(model.jnt_range[left, 0], model.jnt_range[left, 1])
        data.qpos[model.jnt_qposadr[left]] = angles
        data.qpos[model.jnt_qposadr[right]] = angles
    mujoco.mj_kinematics(model, data)
    left_tip = data.body("leg_L4_tarsus").xpos
    right_tip = data.body("leg_R4_tarsus").xpos
    np.testing.assert_allclose(left_tip * [1, -1, 1], right_tip, atol=1e-12)


def test_registry():
    assert set(TASKS) <= set(biorobot.available())


# ----------------------------------------------------------------------------------------------------------------------
# MJC
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("task", TASKS)
def test_mjc_episode(task):
    env = make(task, simulation_time=1.0)
    state, rewards, _ = rollout(env, random_policy(env))
    assert state.truncated
    assert np.abs(rewards).sum() > 1e-3
    assert set(state.observations) == set(env.observation_space.spaces)
    for name, space in env.observation_space.spaces.items():
        observation = np.asarray(state.observations[name], dtype=np.float64)
        assert observation.shape == space.shape, name
        assert np.all(np.isfinite(observation)), name
        assert space.contains(observation), name


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("position_control", [True, False])
@pytest.mark.parametrize("make_policy", [random_policy, bang_bang_policy])
def test_mjc_stable_under_random_actions(task, position_control, make_policy):
    """Uniformly random and bang-bang actions for full default-length episodes never make the simulation diverge.

    The episodes continue past early termination (landing, leaving the track, falling off a platform) to cover the
    full time.
    """
    specification = default_jumping_spider_morphology_specification(
        position_control=position_control
    )
    env = make(task, morphology_specification=specification)
    for seed in range(5):
        policy = make_policy(env, seed)
        state = env.reset(seed)
        for _ in range(env.config.total_num_control_steps):
            state = env.step(state, policy(state))
            assert state.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0
            assert_stable(state.data.qvel)
        assert all(np.all(np.isfinite(o)) for o in state.observations.values())


@pytest.mark.parametrize("noise", [0.0, 0.3])
def test_reset_above_ground(noise):
    """With joint noise, the spider is lifted so that it starts just above the ground instead of inside it."""
    env = make("jumping_spider/long_jump", joint_randomization_noise_scale=noise)
    for seed in range(5):
        state = env.reset(seed)
        distances = [state.data.contact[i].dist for i in range(state.data.ncon)]
        assert min(distances, default=np.inf) >= 0.0
        assert env._lowest_point(state.data) == pytest.approx(0.0005, abs=1e-9)


def test_scripted_jump():
    """The open-loop jump gets the spider fully airborne and lands it upright more than 3 body lengths further."""
    env = biorobot.make("jumping_spider/long_jump")
    state, rewards, max_flight_time = rollout(env, scripted_policy(env))
    distance = state.observations["x_distance"][0]
    assert state.terminated  # landed
    assert (
        max_flight_time >= 0.3
    )  # no contact at all between the spider and the ground for 0.3 s
    assert distance > 3 * BODY_LENGTH
    np.testing.assert_allclose(rewards.sum(), distance, atol=1e-6)
    roll, pitch, _ = state.observations["cephalothorax_rotation"]
    assert np.cos(roll) * np.cos(pitch) > 0.8  # upright


def test_platform_jump_terminates_on_end_platform():
    env = biorobot.make("jumping_spider/platform_jump")
    state, rewards, _ = rollout(env, scripted_policy(env), {"gap": 0.2})
    assert state.terminated
    assert np.all(state.observations["leg_tip_contact"] == 1)
    assert rewards[-1] > env.config.success_bonus
    assert rewards.sum() > env.config.success_bonus + 0.3


def test_platform_jump_terminates_when_falling():
    """Falling short is a terminal failure that earns less than a successful jump."""
    env = biorobot.make("jumping_spider/platform_jump")
    # The jump falls short of an end platform that is 0.5 m away.
    state, rewards, _ = rollout(env, scripted_policy(env), {"gap": 0.5})
    assert state.terminated and not state.truncated
    assert state.observations["cephalothorax_position"][2] < -env.config.fall_depth
    assert rewards.sum() < env.config.success_bonus


def test_directed_jump_terminates_at_target():
    env = biorobot.make("jumping_spider/directed_jump", simulation_time=1.0)
    trajectory = []
    state = env.reset(0, target_position=[5.0, 0.0, 0.0])
    while not state.truncated:
        state = env.step(state, scripted_policy(env)(state))
        trajectory.append(state.observations["cephalothorax_position"].copy())
    # Place the target on the jump's trajectory, at the apex.
    target = trajectory[int(np.argmax([p[2] for p in trajectory]))]
    state, rewards, _ = rollout(env, scripted_policy(env), {"target_position": target})
    assert state.terminated and state.info["has_jumped"]
    assert rewards.sum() > env.config.success_bonus + 0.1


def test_directed_jump_requires_a_jump():
    """Reaching the target without jumping earns no bonus and does not end the episode."""
    env = biorobot.make("jumping_spider/directed_jump")
    state = env.reset(0, target_position=[0.0, 0.0, env._start_height])
    state = env.step(state, np.zeros(env.mj_model.nu))
    assert state.observations["distance_to_target"][0] < env._target_radius
    assert not state.terminated
    assert state.reward < env.config.success_bonus


def test_dragline():
    specification = default_jumping_spider_morphology_specification(dragline=True)
    env = make("jumping_spider/long_jump", morphology_specification=specification)
    assert env.mj_model.ntendon == 1
    assert env.actuator_names[-1].endswith("dragline_brake")
    assert {"dragline_length", "dragline_velocity", "dragline_brake_force"} <= set(
        env.observation_space.spaces
    )
    brake = np.zeros(env.mj_model.nu)
    brake[-1] = 1.0
    state = env.reset(0)
    length = state.observations["dragline_length"][0]
    for _ in range(20):
        state = env.step(state, scripted_jump_action(0.6, env.actuator_names) + brake)
    # The brake can only pull back (negative force) while the spider pays out silk.
    assert state.observations["dragline_brake_force"][0] <= 0.0
    assert state.observations["dragline_length"][0] != length


@pytest.mark.parametrize("task", TASKS)
def test_render_rgb_array(task):
    env = make(task, render_mode="rgb_array", render_size=(48, 64), camera_ids=(0, 2))
    env.config.color_contacts = True
    frames = env.render(env.reset(0))
    assert len(frames) == 2
    assert all(frame.shape == (48, 64, 3) for frame in frames)
    env.close()


# ----------------------------------------------------------------------------------------------------------------------
# MJX
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("task", TASKS)
def test_mjx_matches_mjc(task):
    jax = pytest.importorskip("jax")
    options = FIXED_RESET_OPTIONS[task]
    env_mjc, env_mjx = make(task), make(task, backend="mjx")
    state_mjc = env_mjc.reset(0, **options)
    state_mjx = jax.jit(env_mjx.reset)(jax.random.PRNGKey(0), **options)
    step = jax.jit(env_mjx.step)

    compared = [
        name
        for name in state_mjc.observations
        if "velocity" not in name and "force" not in name
    ]
    for _ in range(10):
        action = scripted_jump_action(
            float(state_mjc.info["time"]), env_mjc.actuator_names
        )
        state_mjc = env_mjc.step(state_mjc, action)
        state_mjx = step(state_mjx, jax.numpy.asarray(action))
        for name in compared:
            np.testing.assert_allclose(
                state_mjx.observations[name],
                state_mjc.observations[name],
                atol=1e-3,
                err_msg=name,
            )
        np.testing.assert_allclose(state_mjx.reward, state_mjc.reward, atol=1e-3)

    # Batched environments with random actions (each with its own target / gap).
    keys = jax.random.split(jax.random.PRNGKey(1), 2)
    batch = jax.vmap(env_mjx.reset)(keys)
    batched_step = jax.jit(jax.vmap(env_mjx.step))
    rng = np.random.default_rng(0)
    for _ in range(5):
        actions = rng.uniform(
            env_mjx.action_low, env_mjx.action_high, size=(2, env_mjx.mj_model.nu)
        )
        batch = batched_step(batch, jax.numpy.asarray(actions))
    for name, space in env_mjx.observation_space.spaces.items():
        assert batch.observations[name].shape == (2, *space.shape)
        assert np.all(np.isfinite(batch.observations[name]))
    if task != "jumping_spider/long_jump":
        mocap = np.asarray(batch.data.mocap_pos)
        assert not np.allclose(mocap[0], mocap[1])


@pytest.mark.slow
def test_mjx_stable_under_random_actions():
    """Uniformly random (first half of the batch) and bang-bang actions (second half) stay stable in float32."""
    jax = pytest.importorskip("jax")
    env = make("jumping_spider/long_jump", backend="mjx")
    batch = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 8))
    batched_step = jax.jit(jax.vmap(env.step))
    rng = np.random.default_rng(0)
    for _ in range(env.config.total_num_control_steps):
        uniform = rng.uniform(
            env.action_low, env.action_high, size=(4, env.mj_model.nu)
        )
        bang_bang = np.where(
            rng.random((4, env.mj_model.nu)) < 0.5, env.action_low, env.action_high
        )
        actions = np.concatenate([uniform, bang_bang])
        batch = batched_step(batch, jax.numpy.asarray(actions))
        assert_stable(batch.data.qvel)
    assert np.all(np.isfinite(batch.data.qpos))


@pytest.mark.slow
def test_mjx_scripted_jump():
    jax = pytest.importorskip("jax")
    env = biorobot.make("jumping_spider/long_jump", backend="mjx")
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    while not (state.terminated or state.truncated):
        action = scripted_jump_action(float(state.info["time"]), env.actuator_names)
        state = step(state, jax.numpy.asarray(action))
    assert state.terminated
    assert state.observations["x_distance"][0] > 3 * BODY_LENGTH
