"""Shared command line and rollout loop of the jumping spider examples."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Optional

import numpy as np

import biorobot
from biorobot.core.env import BRTEnv, EnvState

#: ``policy(env, state, rng) -> action``
Policy = Callable[[BRTEnv, EnvState, np.random.Generator], np.ndarray]


def random_policy(env: BRTEnv, state: EnvState, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(env.action_low, env.action_high)


def parse_args(description: Optional[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backend", default="mjc", choices=["mjc", "mjx", "warp"])
    parser.add_argument(
        "--render", default="human", choices=["human", "rgb_array", "none"]
    )
    parser.add_argument("--episodes", type=int, default=3)
    return parser.parse_args()


def run(
    task: str,
    args: argparse.Namespace,
    policy: Policy = random_policy,
    **make_kwargs: Any,
) -> None:
    """Roll out ``policy`` for ``args.episodes`` episodes and print each episode's return."""
    render_mode = None if args.render == "none" else args.render
    env = biorobot.make(
        task, backend=args.backend, render_mode=render_mode, **make_kwargs
    )
    if args.backend == "mjc":
        reset, step, seed = env.reset, env.step, lambda i: i
    else:
        import jax

        reset, step = jax.jit(env.reset), jax.jit(env.step)
        seed = jax.random.PRNGKey

    rng = np.random.default_rng(0)
    for episode in range(args.episodes):
        state = reset(seed(episode))
        episode_return = 0.0
        while not (bool(state.terminated) or bool(state.truncated)):
            state = step(state, policy(env, state, rng))
            episode_return += float(state.reward)
            if render_mode is not None:
                env.render(state)
        outcome = "terminated" if bool(state.terminated) else "truncated"
        print(
            f"Episode {episode}: return {episode_return:.3f} ({outcome} after {float(state.info['time']):.2f} s)"
        )
    env.close()
