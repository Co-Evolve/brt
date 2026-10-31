"""Command line handling and the episode loop shared by the brittle star examples."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, Optional

import numpy as np

from biorobot.core import BRTEnv, EnvState


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backend", choices=("mjc", "mjx", "warp"), default="mjc")
    parser.add_argument(
        "--render",
        choices=("human", "rgb_array", "none"),
        default="human",
        help="'human' shows an OpenCV window, 'rgb_array' renders off-screen, 'none' does not render.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of control steps to run (default: run forever).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def render_mode(args: argparse.Namespace) -> Optional[str]:
    return None if args.render == "none" else args.render


def run(
    env: BRTEnv,
    policy: Callable[[EnvState, Any], Any],
    args: argparse.Namespace,
    reset_options: Optional[Dict[str, Any]] = None,
) -> EnvState:
    """Run ``policy`` and reset the environment when an episode ends.

    ``policy(state, rng) -> action`` is written with ``env.xp`` (so it can be jitted); ``rng`` is a fresh
    ``np.random.Generator`` (mjc) or JAX PRNG key to use with ``env.backend.uniform``.
    """
    reset_options = reset_options or {}
    if env.backend.name == "mjc":
        rng = np.random.default_rng(args.seed)
        reset = lambda rng: env.reset(rng, **reset_options)
        step = lambda state, rng: env.step(state, policy(state, rng))
        next_rng = lambda rng: (rng, rng)
    else:
        import jax

        rng = jax.random.PRNGKey(args.seed)
        reset = jax.jit(lambda rng: env.reset(rng, **reset_options))
        step = jax.jit(lambda state, rng: env.step(state, policy(state, rng)))
        next_rng = jax.random.split

    rng, reset_rng = next_rng(rng)
    state = reset(reset_rng)
    num_steps = 0
    while args.num_steps is None or num_steps < args.num_steps:
        rng, action_rng = next_rng(rng)
        state = step(state, action_rng)
        num_steps += 1
        if env.config.render_mode is not None:
            env.render(state)
        if bool(state.terminated) or bool(state.truncated):
            print(f"Episode finished at t = {float(state.info['time']):.2f} s")
            rng, reset_rng = next_rng(rng)
            state = reset(reset_rng)
    env.close()
    return state
