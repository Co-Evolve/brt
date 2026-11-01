"""Tests of the seahorse tail morphology (against values of the original model) and its open-loop playground."""

import mujoco
import numpy as np
import pytest

import biorobot
from biorobot.seahorse import (
    HMMTendonRoutingSpecification,
    SeahorseMorphology,
    default_hm_routing_specification,
    default_seahorse_morphology_specification,
    default_tail_morphology_specification,
)
from biorobot.seahorse.environment.tail import make_tail_env

PREFIX = "SeahorseMorphology/"


def contraction_force_to_torque_specification():
    """The 2-segment model with two ventral HM beams of the paper's contraction-force-to-torque experiment."""
    specification = default_seahorse_morphology_specification(
        num_segments=2, hm_segment_span=0, p_control=False, mvm_enabled=False
    )
    hm = specification.beam_actuation_specification.hm_beam_actuation_specification
    hm.f_control_gear = 5.0
    hm.routing_specifications = [
        HMMTendonRoutingSpecification(
            start_segment=-1,
            end_segment=1,
            intermediate_points=[0],
            sagittal_sides=["ventral"],
            coronal_sides=["dextral", "sinistral"],
        )
    ]
    return specification


def small_tail_specification():
    """3 segments, force-controlled ventral HM beams over the whole tail and MVM beams."""
    specification = default_seahorse_morphology_specification(
        num_segments=3, hm_segment_span=2, p_control=False, mvm_enabled=True
    )
    specification.beam_actuation_specification.hm_beam_actuation_specification.routing_specifications = [
        default_hm_routing_specification(num_segments=3, segment_span=2)
    ]
    return specification


def ramp_action(env, step, num_steps):
    """Contract every beam linearly from relaxed (upper end of the control range) to fully contracted."""
    alpha = step / num_steps
    return env.action_high + alpha * (env.action_low - env.action_high)


# ----------------------------------------------------------------------------------------------------------------------
# Morphology
# ----------------------------------------------------------------------------------------------------------------------
def test_morphology_counts():
    model = SeahorseMorphology(default_tail_morphology_specification()).compile()
    # 20 bodies per segment; 3 vertebral and 8 gliding joints per segment but the first.
    assert model.nbody == 1 + 11 * 20
    assert model.njnt == 10 * 11
    assert model.ngeom == 11 * 14
    # 4 vertebral struts per segment, 2 HM beams, 2 MVM beams per pair of neighbouring segments.
    assert model.ntendon == 11 * 4 + 2 + 20
    assert model.nu == 2 + 20
    # 8 gliding joint couplings within every segment but the first and 8 between every two of those segments.
    assert model.neq == 10 * 8 + 9 * 8
    # jointpos and jointactuatorfrc per vertebral joint, tendonpos and actuatorfrc per beam.
    assert model.nsensor == 2 * 30 + 2 * 22
    # One mesh asset per file.
    assert model.nmesh == 14


def test_original_model_values():
    """Values of the original model (dm_control MJCF, MuJoCo 3.2.2)."""
    model = SeahorseMorphology(default_tail_morphology_specification()).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mass = lambda body: model.body(body).mass[0]
    assert mass("segment_0_plate_0_plate_body") == pytest.approx(0.01094)
    assert mass("segment_3_plate_2_plate_body") == pytest.approx(0.00971)
    assert mass("segment_5_vertebrae_vertebrae_body") == pytest.approx(0.00452)
    assert mass("segment_5_vertebrae_ball_bearing_body") == pytest.approx(0.00214)
    assert mass("segment_5_plate_1_connector_x_body") == pytest.approx(0.00014004)
    # Body positions in the tail's frame (built along +z).
    np.testing.assert_allclose(
        data.body("segment_3_plate_2_plate_body").xpos,
        [-0.036203867196751234, 0.036203867196751234, 0.1035],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        data.body("segment_10_vertebrae").xpos, [0, 0, 0.32], atol=1e-12
    )
    # Struts rest at their initial length; HM beams of the default (paper's HM) routing.
    strut = model.tendon("segment_4_vertebral_vertebral_strut_ventral")
    assert model.tendon_length0[strut.id] == pytest.approx(0.037)
    assert model.tendon_lengthspring[strut.id, 0] == pytest.approx(0.037)
    assert model.tendon_stiffness[strut.id] == pytest.approx(3.043)
    pitch = model.joint("segment_1_vertebrae_vertebrae_joint_pitch")
    np.testing.assert_allclose(pitch.range, [-14 / 180 * np.pi, 14 / 180 * np.pi])
    assert model.dof_armature[pitch.dofadr[0]] == pytest.approx(0.01)
    assert model.dof_frictionloss[pitch.dofadr[0]] == pytest.approx(0.0015)
    glide = model.joint("segment_1_plate_0_x_axis")
    np.testing.assert_allclose(glide.range, [-0.005, 0.0])
    np.testing.assert_allclose(glide.solimp, [0.9, 0.9999, 0.001, 0.1, 6])
    mvm = model.actuator("mvm_beam_sinistral_(0, 1)")
    np.testing.assert_allclose(mvm.ctrlrange, [0.0, 0.0285], atol=1e-12)
    np.testing.assert_allclose(mvm.biasprm[:3], [0, -1000, 0])
    hm = model.actuator("hm_beam_ventral_dextral_10")
    np.testing.assert_allclose(hm.gear[0], 10.0)
    np.testing.assert_allclose(hm.forcerange, [-10.0, 0.0])


def test_hm_routing():
    """A beam passes the ghost points of its start segment, two points per intermediate segment and its end plate."""
    specification = default_seahorse_morphology_specification(
        num_segments=11, hm_segment_span=10, p_control=False, mvm_enabled=False
    )
    hm = specification.beam_actuation_specification.hm_beam_actuation_specification
    hm.routing_specifications = [
        default_hm_routing_specification(
            num_segments=11, segment_span=10, coronal_sides=["sinistral"]
        )
    ]
    model = SeahorseMorphology(specification).compile()
    assert model.ntendon == 45
    beam = model.tendon("hm_beam_ventral_sinistral_10")
    sites = [
        model.site(model.wrap_objid[model.tendon_adr[beam.id] + i]).name
        for i in range(model.tendon_num[beam.id])
    ]
    assert sites[:2] == [
        "segment_0_plate_1_ghost_hm_tap_a_0",
        "segment_0_plate_1_ghost_hm_tap_a_1",
    ]
    assert sites[2] == "segment_1_plate_1_intermediate_hm_tap_9_0"
    assert sites[-3] == "segment_9_plate_1_intermediate_hm_tap_1_1"
    assert sites[-2:] == [
        "segment_10_plate_1_end_hm_tap_0",
        "segment_10_plate_1_end_hm_tap_1",
    ]
    assert model.tendon_length0[beam.id] == pytest.approx(0.3354536151686237)


def test_registry():
    assert "seahorse/tail" in biorobot.available()


# ----------------------------------------------------------------------------------------------------------------------
# MJC
# ----------------------------------------------------------------------------------------------------------------------
def test_mjc_episode():
    env = biorobot.make("seahorse/tail")
    assert env.mj_model.opt.timestep == pytest.approx(0.004)
    assert env.config.total_num_control_steps == 125
    state = env.reset(0)
    num_steps = 0
    while not state.truncated:
        state = env.step(state, ramp_action(env, num_steps, 125))
        num_steps += 1
        assert state.reward == 0.0 and not state.terminated
        assert state.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0
    assert num_steps == 125
    assert set(state.observations) == set(env.observation_space.spaces)
    for name, space in env.observation_space.spaces.items():
        observation = np.asarray(state.observations[name], dtype=np.float64)
        assert observation.shape == space.shape, name
        assert space.contains(observation), name
    assert env.observation_space["joint_actuator_force"].shape == (30,)
    assert env.observation_space["actuator_force"].shape == (22,)


def test_ventral_bending():
    """Contracting the ventral HM beams bends every vertebral joint ventrally (positive pitch)."""
    env = biorobot.make("seahorse/tail", simulation_time=2.0)
    state = env.reset(0)
    for _ in range(env.config.total_num_control_steps):
        action = env.action_high.copy()
        action[:2] = -1.0  # full force on both HM beams; MVM beams relaxed
        state = env.step(state, action)
    pitch = state.observations["vertebrae_pitch_joint_pos"]
    assert np.all(pitch > 0.1)
    assert np.all(state.observations["joint_actuator_force"][0::3] > 0)


def test_contraction_force_to_torque_rollout():
    """A force ramp on the 2-segment model reproduces the original simulation (MuJoCo 3.2.2) at its last step."""
    env = make_tail_env(
        morphology_specification=contraction_force_to_torque_specification()
    )
    state = env.reset(0)
    for step in range(125):
        state = env.step(state, -step / 125 * np.ones(2))
    # The pitch joint slightly exceeds its (soft) limit of 14 degrees.
    assert state.data.qpos[0] == pytest.approx(0.24504125899693927, rel=1e-5)
    assert state.observations["joint_actuator_force"][0] == pytest.approx(
        0.29000808975912756, rel=1e-5
    )
    # Actuator forces of force-controlled beams are fractions of f_control_gear (the actuator's gear).
    np.testing.assert_allclose(state.observations["actuator_force"], [-0.992, -0.992])


@pytest.mark.parametrize("seed", [0, 1])
def test_mjc_stable_under_random_actions(seed):
    """Force-controlled HM beams and MVM beams stay stable under bang-bang actions at the default 4 ms timestep."""
    env = biorobot.make("seahorse/tail", simulation_time=2.0)
    rng = np.random.default_rng(seed)
    state = env.reset(seed)
    for _ in range(env.config.total_num_control_steps):
        action = np.where(
            rng.random(env.mj_model.nu) < 0.5, env.action_low, env.action_high
        )
        state = env.step(state, action)
        assert state.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0
    assert np.all(np.isfinite(state.data.qpos))


def test_contacts_and_gravity_options():
    env = make_tail_env(
        morphology_specification=contraction_force_to_torque_specification(),
        enable_contacts=True,
        enable_gravity=True,
    )
    assert not env.mj_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT
    assert not env.mj_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY
    assert np.all(env.mj_model.geom_contype == 1)
    env = make_tail_env(
        morphology_specification=contraction_force_to_torque_specification()
    )
    assert env.mj_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT
    assert env.mj_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY
    assert np.all(env.mj_model.geom_contype == 0)


def test_render_rgb_array():
    env = make_tail_env(
        morphology_specification=contraction_force_to_torque_specification(),
        render_mode="rgb_array",
        render_size=(48, 64),
        camera_ids=(0, 2),
    )
    frames = env.render(env.reset(0))
    assert len(frames) == 2
    assert all(frame.shape == (48, 64, 3) for frame in frames)
    env.close()


# ----------------------------------------------------------------------------------------------------------------------
# MJX
# ----------------------------------------------------------------------------------------------------------------------
def test_mjx_matches_mjc():
    jax = pytest.importorskip("jax")
    specification = small_tail_specification()
    env_mjc = make_tail_env(morphology_specification=specification)
    env_mjx = make_tail_env(backend="mjx", morphology_specification=specification)
    state_mjc = env_mjc.reset(0)
    state_mjx = jax.jit(env_mjx.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env_mjx.step)
    for i in range(10):
        action = ramp_action(env_mjc, i, 10)
        state_mjc = env_mjc.step(state_mjc, action)
        state_mjx = step(state_mjx, jax.numpy.asarray(action))
    for name in state_mjc.observations:
        np.testing.assert_allclose(
            state_mjx.observations[name],
            state_mjc.observations[name],
            rtol=1e-2,
            atol=1e-3,
            err_msg=name,
        )
    assert float(state_mjx.reward) == 0.0


@pytest.mark.slow
def test_mjx_default_tail_episode():
    jax = pytest.importorskip("jax")
    env_mjc = biorobot.make("seahorse/tail")
    env_mjx = biorobot.make("seahorse/tail", backend="mjx")
    state_mjc = env_mjc.reset(0)
    state_mjx = jax.jit(env_mjx.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env_mjx.step)
    num_steps = env_mjc.config.total_num_control_steps
    for i in range(num_steps):
        action = ramp_action(env_mjc, i, num_steps)
        state_mjc = env_mjc.step(state_mjc, action)
        state_mjx = step(state_mjx, jax.numpy.asarray(action))
    assert bool(state_mjx.truncated)
    np.testing.assert_allclose(
        state_mjx.observations["vertebrae_pitch_joint_pos"],
        state_mjc.observations["vertebrae_pitch_joint_pos"],
        atol=1e-2,
    )
