import numpy as np

from biorobot.jumping_spider.environment.directed_jump.mjc_env import JumpingSpiderDirectedJumpMJCEnvironment, \
    JumpingSpiderDirectedJumpEnvironmentConfiguration
from biorobot.jumping_spider.mjcf.arena.directed_jump import DirectedJumpArenaConfiguration, MJCFDirectedJumpArena
from biorobot.jumping_spider.mjcf.morphology.morphology import MJCFJumpingSpiderMorphology
from biorobot.jumping_spider.mjcf.morphology.specification.default import default_jumping_spider_specification
from biorobot.jumping_spider.mjcf.morphology.specification.specification import JumpingSpiderMorphologySpecification
from biorobot.jumping_spider.usage_examples.platform_jump import post_render


def create_morphology_specification() -> JumpingSpiderMorphologySpecification:
    morphology_specification = default_jumping_spider_specification(dragline=False, position_control=True)

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


if __name__ == "__main__":
    RENDER_MODE = "human"

    env = create_env(render_mode=RENDER_MODE)

    env_rng = np.random.RandomState(0)
    step_fn = env.step
    reset_fn = env.reset

    state = reset_fn(env_rng)
    while True:
        action = env.action_space.sample()
        state = step_fn(state=state, action=action)
        post_render(env.render(state=state), env.environment_configuration)
        if state.terminated | state.truncated:
            state = reset_fn(env_rng)
    env.close()
