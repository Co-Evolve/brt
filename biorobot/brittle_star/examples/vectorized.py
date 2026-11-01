"""Many light escape episodes in parallel with MJX or MuJoCo Warp: ``jax.vmap`` over reset and step."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

import biorobot

NUM_ENVS = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mjx", "warp"), default="mjx")
    parser.add_argument(
        "--render", choices=("human", "rgb_array", "none"), default="human"
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    args = parser.parse_args()

    env = biorobot.make(
        "brittle_star/light_escape",
        backend=args.backend,
        light_perlin_noise_scale=10,
        simulation_time=5,
        render_mode=None if args.render == "none" else args.render,
        render_size=(240, 320),
        camera_ids=(0, 1),
        color_contacts=True,
    )
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    @jax.jit
    def sample_actions(rng):
        rng, sub_rng = jax.random.split(rng)
        actions = jax.random.uniform(
            sub_rng,
            (args.num_envs, env.mj_model.nu),
            minval=env.action_low,
            maxval=env.action_high,
        )
        return actions, rng

    rng, reset_rng = jax.random.split(jax.random.PRNGKey(0))
    state = reset(jax.random.split(reset_rng, args.num_envs))
    num_steps = 0
    while args.num_steps is None or num_steps < args.num_steps:
        actions, rng = sample_actions(rng)
        state = step(state, actions)
        num_steps += 1
        if env.config.render_mode is not None:
            env.render(state)  # One row of camera views per environment.
        if bool(jnp.all(state.truncated)):
            rng, reset_rng = jax.random.split(rng)
            state = reset(jax.random.split(reset_rng, args.num_envs))
    print("Mean reward of the last step:", float(state.reward.mean()))
    env.close()


if __name__ == "__main__":
    main()
