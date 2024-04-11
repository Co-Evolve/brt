from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random
import numpy as np
from moojoco.environment.mjc_env import MJCEnvState
from moojoco.environment.mjx_env import MJXEnvState

from biorobot.brittle_star.environment.undirected_locomotion.dual import (
    BrittleStarUndirectedLocomotionEnvironment,
)
from biorobot.brittle_star.environment.undirected_locomotion.shared import (
    BrittleStarUndirectedLocomotionEnvironmentConfiguration,
)
from biorobot.brittle_star.mjcf.arena.aquarium import (
    AquariumArenaConfiguration,
    MJCFAquariumArena,
)
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import (
    default_brittle_star_morphology_specification,
)


def create_env(
    backend: str, render_mode: str
) -> BrittleStarUndirectedLocomotionEnvironment:
    morphology_spec = default_brittle_star_morphology_specification(
        num_arms=5,
        num_segments_per_arm=6,
        use_p_control=True,
        use_torque_control=False,
        radius_to_strength_factor=200,
    )
    morphology = MJCFBrittleStarMorphology(morphology_spec)
    arena_config = AquariumArenaConfiguration()
    arena = MJCFAquariumArena(configuration=arena_config)

    env_config = BrittleStarUndirectedLocomotionEnvironmentConfiguration(
        render_mode=render_mode,
        num_physics_steps_per_control_step=10,
        simulation_time=5,
        time_scale=1,
        camera_ids=[0, 1],
        color_contacts=True,
        solver_iterations=1,
        solver_ls_iterations=5,
    )
    env = BrittleStarUndirectedLocomotionEnvironment.from_morphology_and_arena(
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

        def action_sample_fn(_: None, state: MJCEnvState) -> Tuple[np.ndarray, None]:
            return env.action_space.sample(), None

    else:
        env_rng, action_rng = jax.random.split(jax.random.PRNGKey(0), 2)

        step_fn = jax.jit(env.step)
        reset_fn = jax.jit(env.reset)

        def action_sample_fn(
            rng: chex.PRNGKey, state: MJXEnvState
        ) -> Tuple[jnp.ndarray, chex.PRNGKey]:
            time = state.info["time"]
            actions = jnp.zeros(env.action_space.shape).reshape(5, -1)
            actions = actions.at[0, 1::2].set(1)
            actions = actions.at[1:5, 1::2].set(jnp.sin(5 * time))
            actions = actions.at[1:5, ::2].set(jnp.cos(5 * time))
            actions = actions.at[1:3, ::2].set(-1 * actions[1:3, ::2])
            actions = actions.flatten()
            return actions, rng

    state = reset_fn(env_rng)
    while True:
        action, action_rng = action_sample_fn(action_rng, state)
        state = step_fn(state=state, action=action)
        print(state.observations["joint_position"])
        env.render(state=state)
    env.close()
