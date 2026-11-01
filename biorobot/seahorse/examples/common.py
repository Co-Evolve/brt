"""Shared command line and open-loop rollout of the seahorse examples."""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Optional

import numpy as np

from biorobot.core.env import BRTEnv

#: ``controller(env, t) -> action``: an open-loop controller, a function of the time ``t`` (s) only.
OpenLoopController = Callable[[BRTEnv, float], np.ndarray]


def parse_args(description: Optional[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", default="mjc", choices=["mjc", "mjx", "warp"])
    parser.add_argument(
        "--render", default="human", choices=["human", "rgb_array", "none"]
    )
    return parser.parse_args()


def rollout(env: BRTEnv, controller: OpenLoopController) -> List[Dict[str, np.ndarray]]:
    """Run one episode with ``controller`` (rendering every step if the environment has a render mode).

    Returns the observations after every control step.
    """
    if env.backend.name == "mjc":
        state, step = env.reset(0), env.step
    else:
        import jax

        state, step = jax.jit(env.reset)(jax.random.PRNGKey(0)), jax.jit(env.step)
    observations = []
    while not bool(state.truncated):
        state = step(state, controller(env, float(state.info["time"])))
        observations.append({k: np.asarray(v) for k, v in state.observations.items()})
        if env.config.render_mode is not None:
            env.render(state)
    return observations


def ramp(t: float, start: float, duration: float) -> float:
    """0 before ``start``, rising linearly to 1 over ``duration`` seconds, 1 afterwards."""
    return float(np.clip((t - start) / duration, 0.0, 1.0))


def contraction(env: BRTEnv, level: np.ndarray) -> np.ndarray:
    """Actions that contract every beam by ``level`` (0: relaxed, 1: fully contracted; per actuator or scalar).

    Every beam's actuator pulls more towards the lower end of its control range (shorter target length under position
    control, a larger force under force control), so the relaxed action is the upper end of the range.
    """
    return env.action_high + level * (env.action_low - env.action_high)
