from typing import Tuple

import numpy as np

from biorobot.jumping_spider.environment.mjc_env import JumpingSpiderMJCEnvironment, \
    JumpingSpiderEnvironmentConfiguration
from biorobot.jumping_spider.mjcf.arena.platform_jump import PlatformJumpArenaConfiguration, MJCFPlatformJumpArena
from biorobot.jumping_spider.mjcf.morphology.morphology import MJCFJumpingSpiderMorphology
from biorobot.jumping_spider.mjcf.morphology.specification.default import default_jumping_spider_specification


def create_env() -> JumpingSpiderMJCEnvironment:
    morphology_spec = default_jumping_spider_specification()
    morphology = MJCFJumpingSpiderMorphology(specification=morphology_spec)

    arena_config = PlatformJumpArenaConfiguration()
    arena = MJCFPlatformJumpArena(configuration=arena_config)

    env_config = JumpingSpiderEnvironmentConfiguration(
        num_physics_steps_per_control_step=10,
        simulation_time=5,
        time_scale=1,
        camera_ids=[0, 1, 2],
        render_mode="human"
    )

    env = JumpingSpiderMJCEnvironment.from_morphology_and_arena(
        morphology=morphology, arena=arena, configuration=env_config
    )
    return env


if __name__ == "__main__":
    env = create_env()

    env_rng, action_rng = np.random.RandomState(0), None
    step_fn = env.step
    reset_fn = env.reset


    def action_sample_fn(_: None) -> Tuple[np.ndarray, None]:
        return env.action_space.sample(), None


    state = reset_fn(env_rng, [-1.0, 0.0, 0.05])
    while True:
        action, action_rng = action_sample_fn(action_rng)
        state = step_fn(state=state, action=action)
        env.render(state=state)
        if state.terminated | state.truncated:
            state = reset_fn(env_rng)
    env.close()
