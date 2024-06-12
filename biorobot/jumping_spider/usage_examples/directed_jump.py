import numpy as np
from fprs.parameters import FixedParameter

from biorobot.jumping_spider.environment.directed_jump.mjc_env import JumpingSpiderDirectedJumpMJCEnvironment, \
    JumpingSpiderDirectedJumpEnvironmentConfiguration
from biorobot.jumping_spider.mjcf.arena.directed_jump import DirectedJumpArenaConfiguration, MJCFDirectedJumpArena
from biorobot.jumping_spider.mjcf.morphology.morphology import MJCFJumpingSpiderMorphology
from biorobot.jumping_spider.mjcf.morphology.specification.default import default_jumping_spider_specification
from biorobot.jumping_spider.mjcf.morphology.specification.specification import JumpingSpiderMorphologySpecification
from biorobot.jumping_spider.usage_examples.platform_jump import post_render
import mujoco

def create_morphology_specification() -> JumpingSpiderMorphologySpecification:
    morphology_specification = default_jumping_spider_specification(dragline=False, position_control=True)
    morphology_specification.abdomen_specification.torque_limit = FixedParameter(50)
    for leg_specification in morphology_specification.leg_specifications:
        for segment_specification in leg_specification.segment_specifications:
            segment_specification.torque_limit = FixedParameter(50)

    return morphology_specification


def create_env(
        render_mode: str
) -> JumpingSpiderDirectedJumpMJCEnvironment:
    morphology_specification = create_morphology_specification()
    morphology = MJCFJumpingSpiderMorphology(specification=morphology_specification)
    arena_configuration = DirectedJumpArenaConfiguration()
    arena = MJCFDirectedJumpArena(configuration=arena_configuration)

    env_config = JumpingSpiderDirectedJumpEnvironmentConfiguration(
        render_mode=render_mode,
        num_physics_steps_per_control_step=10,
        simulation_time=5,
        time_scale=1,
        camera_ids=[0, 1],
        color_contacts=True,
        target_distance_range=(10, 15),
        target_angle_range=(20 / 180 * np.pi, 60 / 180 * np.pi)
    )
    env = JumpingSpiderDirectedJumpMJCEnvironment.from_morphology_and_arena(
        morphology=morphology, arena=arena, configuration=env_config
    )
    return env

def get_q0(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    joint_qpos_adr = ([model.joint(joint_id).qposadr[0] for joint_id in range(model.njnt) if
                       "abdomen" in model.joint(joint_id).name] +
                      [model.joint(joint_id).qposadr[0] for joint_id in range(model.njnt) if
                       "leg" in model.joint(joint_id).name])

    return model.qpos0[joint_qpos_adr]

if __name__ == "__main__":
    RENDER_MODE = "human"

    env = create_env(render_mode=RENDER_MODE)
    # oop_mask = ["out_of_plane" in actuator and "leg_3" in actuator and "pat" in actuator for actuator in env.actuators]
    oop_mask = ["out_of_plane" in actuator and "pat" in actuator for actuator in env.actuators]
    env_rng = np.random.RandomState(0)
    step_fn = env.step
    reset_fn = env.reset

    state = reset_fn(env_rng)

    while True:
        ctrl = get_q0(model=state.mj_model, data=state.mj_data)

        if state.info["time"] > 2:
            ctrl[oop_mask] = env.action_space.high[oop_mask]

        # action = env.action_space.sample() * 0
        # state = step_fn(state=state, action=action)
        state = step_fn(state=state, action=ctrl)
        post_render(env.render(state=state), env.environment_configuration)


        if state.terminated | state.truncated:
            state = reset_fn(env_rng)
    env.close()
