import numpy as np

from biorobot.jumping_spider.environment.long_jump.mjc_env import JumpingSpiderLongJumpEnvironmentConfiguration, \
    JumpingSpiderLongJumpMJCEnvironment
from biorobot.jumping_spider.mjcf.arena.long_jump import LongJumpArenaConfiguration, MJCFLongJumpArena
from biorobot.jumping_spider.mjcf.morphology.morphology import MJCFJumpingSpiderMorphology
from biorobot.jumping_spider.mjcf.morphology.specification.default import default_jumping_spider_specification
from biorobot.jumping_spider.usage_examples.directed_jump import post_render


def create_env(
        render_mode: str
) -> JumpingSpiderLongJumpMJCEnvironment:
    morphology_specification = default_jumping_spider_specification(dragline=True, position_control=False)
    morphology = MJCFJumpingSpiderMorphology(specification=morphology_specification)

    arena_configuration = LongJumpArenaConfiguration()
    arena = MJCFLongJumpArena(configuration=arena_configuration)

    env_config = JumpingSpiderLongJumpEnvironmentConfiguration(
        render_mode=render_mode,
        num_physics_steps_per_control_step=10,
        simulation_time=5,
        time_scale=1,
        camera_ids=[0, 1],
        color_contacts=True,
    )
    env = JumpingSpiderLongJumpMJCEnvironment.from_morphology_and_arena(
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
