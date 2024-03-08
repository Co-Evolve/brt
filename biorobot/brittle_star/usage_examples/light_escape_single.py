from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random
import numpy as np

from biorobot.brittle_star.environment.light_escape.dual import (
    BrittleStarLightEscapeEnvironment,
)
from biorobot.brittle_star.environment.light_escape.shared import (
    BrittleStarLightEscapeEnvironmentConfiguration,
)
from biorobot.brittle_star.mjcf.arena.aquarium import (
    AquariumArenaConfiguration,
    MJCFAquariumArena,
)
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import (
    default_brittle_star_morphology_specification,
)


def create_env(backend: str, render_mode: str) -> BrittleStarLightEscapeEnvironment:
    morphology_spec = default_brittle_star_morphology_specification(
        num_arms=5, num_segments_per_arm=5, use_p_control=True
    )
    morphology = MJCFBrittleStarMorphology(morphology_spec)
    arena_config = AquariumArenaConfiguration(sand_ground_color=True)
    arena = MJCFAquariumArena(configuration=arena_config)

    env_config = BrittleStarLightEscapeEnvironmentConfiguration(
        render_mode=render_mode,
        light_perlin_noise_scale=10,
        num_physics_steps_per_control_step=10,
        simulation_time=5,
        time_scale=1,
        camera_ids=[0, 1],
        color_contacts=True,
    )
    env = BrittleStarLightEscapeEnvironment.from_morphology_and_arena(
        morphology=morphology, arena=arena, configuration=env_config, backend=backend
    )
    return env


if __name__ == "__main__":
    BACKEND = "MJX"
    RENDER_MODE = "human"

    env = create_env(backend=BACKEND, render_mode=RENDER_MODE)

    if BACKEND == "MJC":
        env_rng, action_rng = np.random.RandomState(0), None
        step_fn = env.step
        reset_fn = env.reset

        def action_sample_fn(_: None) -> Tuple[np.ndarray, None]:
            return env.action_space.sample(), None

    else:
        env_rng, action_rng = jax.random.split(jax.random.PRNGKey(0), 2)

        step_fn = jax.jit(env.step)
        reset_fn = jax.jit(env.reset)

        def action_sample_fn(rng: chex.PRNGKey) -> Tuple[jnp.ndarray, chex.PRNGKey]:
            rng, sub_rng = jax.random.split(rng, 2)
            return env.action_space.sample(rng=sub_rng), rng

    state = reset_fn(env_rng)
    while True:
        action, action_rng = action_sample_fn(action_rng)
        state = step_fn(state=state, action=action)
        print(state.observations["joint_position"])
        print(state.observations["joint_velocity"])
        print(state.observations["joint_actuator_force"])
        print()
        env.render(state=state)
    env.close()
