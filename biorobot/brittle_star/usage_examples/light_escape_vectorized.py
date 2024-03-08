from functools import partial
from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random
import numpy as np
from moojoco.environment.mjc_env import ThreadedVectorMJCEnvWrapper

from biorobot.brittle_star.usage_examples.light_escape_single import create_env
from biorobot.toy_example.usage_examples.mjc.example_usage_single import post_render

if __name__ == "__main__":
    BACKEND = "MJX"
    RENDER_MODE = "rgb_array"
    NUM_ENVS = 2

    if BACKEND == "MJC":
        env_rng, action_rng = [np.random.RandomState(i) for i in range(NUM_ENVS)], None

        env = ThreadedVectorMJCEnvWrapper(
            create_env_fn=partial(create_env, backend=BACKEND, render_mode=RENDER_MODE),
            num_environments=NUM_ENVS,
        )

        step_fn = env.step
        reset_fn = env.reset

        def action_sample_fn(_: None) -> Tuple[np.ndarray, None]:
            return env.action_space.sample(), None

    else:
        action_rng, env_rng = jax.random.split(jax.random.PRNGKey(0), 2)
        action_rng = jnp.array(jax.random.split(action_rng, NUM_ENVS))
        env_rng = jnp.array(jax.random.split(env_rng, NUM_ENVS))

        env = create_env(backend=BACKEND, render_mode=RENDER_MODE)

        step_fn = jax.jit(jax.vmap(env.step))
        reset_fn = jax.jit(jax.vmap(env.reset))

        def action_sample_fn(rng: chex.PRNGKey) -> Tuple[jnp.ndarray, chex.PRNGKey]:
            rng, sub_rng = jax.random.split(rng, 2)
            return env.action_space.sample(rng=sub_rng), rng

        action_sample_fn = jax.jit(jax.vmap(action_sample_fn))

    state = reset_fn(env_rng)

    while True:
        action, action_rng = action_sample_fn(action_rng)
        state = step_fn(state=state, action=action)
        post_render(
            env.render(state=state),
            environment_configuration=env.environment_configuration,
        )
    env.close()
