from typing import Callable

import jax
import jax.numpy as jnp
from moojoco.environment import mjx_spaces

from biorobot.toy_example.environment.mjc_env import ToyExampleEnvironmentConfiguration
from biorobot.toy_example.environment.mjx_env import ToyExampleMJXEnvironment
from biorobot.toy_example.mjcf.arena.arena import (
    MJCFPlaneWithTargetArena,
    PlaneWithTargetArenaConfiguration,
)
from biorobot.toy_example.mjcf.morphology.morphology import MJCFToyExampleMorphology
from biorobot.toy_example.mjcf.morphology.specification.default import (
    default_toy_example_morphology_specification,
)
from biorobot.toy_example.usage_examples.mjc.example_usage_single import post_render


def create_mjx_environment(
    environment_configuration: ToyExampleEnvironmentConfiguration,
) -> ToyExampleMJXEnvironment:
    morphology_specification = default_toy_example_morphology_specification(
        num_arms=4, num_segments_per_arm=2
    )
    morphology = MJCFToyExampleMorphology(specification=morphology_specification)
    arena_configuration = PlaneWithTargetArenaConfiguration()
    arena = MJCFPlaneWithTargetArena(configuration=arena_configuration)
    env = ToyExampleMJXEnvironment.from_morphology_and_arena(
        morphology=morphology, arena=arena, configuration=environment_configuration
    )
    return env


def create_mjx_open_loop_controller(
    single_action_space: mjx_spaces.Box, num_envs: int
) -> Callable[[float], jnp.ndarray]:
    def open_loop_controller(t: float) -> jnp.ndarray:
        actions = jnp.ones(single_action_space.shape)
        actions = actions.at[jnp.arange(0, len(actions), 2)].set(jnp.cos(5 * t))
        actions = actions.at[jnp.arange(1, len(actions), 2)].set(jnp.sin(5 * t))
        actions = actions.at[jnp.arange(len(actions) // 2, len(actions), 2)].set(
            actions[jnp.arange(len(actions) // 2, len(actions), 2)] * -1
        )
        return actions

    if num_envs > 1:
        open_loop_controller = jax.vmap(open_loop_controller)

    open_loop_controller = jax.jit(open_loop_controller)

    return open_loop_controller


if __name__ == "__main__":
    environment_configuration = ToyExampleEnvironmentConfiguration(
        render_mode="human", camera_ids=[0, 1]
    )
    env = create_mjx_environment(environment_configuration=environment_configuration)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng=rng)

    controller = create_mjx_open_loop_controller(
        single_action_space=env.action_space, num_envs=1
    )

    steps = 0
    fps = 30
    while not (state.truncated | state.terminated):
        t = state.info["time"]
        actions = controller(t)

        state = jit_step(state=state, action=actions)
        if steps % int((1 / fps) / environment_configuration.control_timestep) == 0:
            post_render(
                env.render(state), environment_configuration=environment_configuration
            )
            print(state.observations["segment_ground_contact"])

        steps += 1
    env.close()
