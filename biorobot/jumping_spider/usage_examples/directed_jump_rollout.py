import mujoco
import numpy as np

from mujoco import rollout
from biorobot.jumping_spider.usage_examples.directed_jump import create_env
from biorobot.jumping_spider.usage_examples.platform_jump import post_render


def get_q0(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    joint_qpos_adr = ([model.joint(joint_id).qposadr[0] for joint_id in range(model.njnt) if
                       "abdomen" in model.joint(joint_id).name] +
                      [model.joint(joint_id).qposadr[0] for joint_id in range(model.njnt) if
                       "leg" in model.joint(joint_id).name])

    return model.qpos0[joint_qpos_adr]


def set_initial_pose(model: mujoco.MjModel, data: mujoco.MjData, leg_joint_positions: np.ndarray | None,
                     abdomen_joint_positions: np.ndarray | None) -> mujoco.MjData:
    leg_joints_qpos_adrs = np.array(
        [model.joint(joint_id).qposadr[0] for joint_id in range(model.njnt) if "leg" in model.joint(joint_id).name])
    abdomen_joints_qpos_adrs = np.array(
        [model.joint(joint_id).qposadr[0] for joint_id in range(model.njnt) if "abdomen" in model.joint(joint_id).name]
    )

    if leg_joint_positions is None:
        data.qpos[leg_joints_qpos_adrs] = model.qpos0[leg_joints_qpos_adrs]
    else:
        assert len(leg_joint_positions) == len(
            leg_joints_qpos_adrs), (f"Mismatch between actual number of leg joints ({len(leg_joints_qpos_adrs)}) "
                                    f"and given amount of joint positions ({len(leg_joint_positions)})")
        data.qpos[leg_joints_qpos_adrs] = leg_joint_positions

    if abdomen_joint_positions is None:
        data.qpos[abdomen_joints_qpos_adrs] = model.qpos0[abdomen_joints_qpos_adrs]
    else:
        assert len(abdomen_joint_positions) == len(
            abdomen_joints_qpos_adrs), (
            f"Mismatch between actual number of abdomen joints ({len(abdomen_joints_qpos_adrs)}) "
            f"and given amount of joint positions ({len(abdomen_joint_positions)})")
        data.qpos[leg_joints_qpos_adrs] = abdomen_joint_positions

    return data


if __name__ == "__main__":
    RENDER_MODE = "human"

    env = create_env(render_mode=RENDER_MODE)

    rng = np.random.RandomState(0)
    mjc_env_state = env.reset(rng=rng)

    model = mjc_env_state.mj_model
    data = mjc_env_state.mj_data
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    data = set_initial_pose(model=model, data=data, leg_joint_positions=None, abdomen_joint_positions=None)

    nroll = 1  # number of initial states
    nstep = env.environment_configuration.total_num_physics_steps

    initial_state = np.empty(nstate)
    mujoco.mj_getState(m=model, d=data, state=initial_state, spec=mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial_state = np.repeat(initial_state, nroll).reshape(nroll, nstate)

    control = np.random.randn(nroll, nstep, model.nu) * 0
    control[:, :, :] = get_q0(model=model, data=data)

    states, sensordata = rollout.rollout(model, data, initial_state, control)

    for rollout in states:
        for timestep in rollout:
            timestep_2d = timestep.reshape((nstate, 1)).astype(np.float64)
            mujoco.mj_setState(model, data, timestep_2d, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mjc_env_state = mjc_env_state.replace(mj_model=model, mj_data=data)
            post_render(render_output=env.render(state=mjc_env_state),
                        environment_configuration=env.environment_configuration)
            mjc_env_state = env._update_observations(state=mjc_env_state)
            mjc_env_state = env._update_info(state=mjc_env_state)
            print(mjc_env_state.info["time"])
        break
