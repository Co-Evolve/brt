"""Gymnasium adapter for the ``mjc`` backend, for use with standard RL libraries."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np

from biorobot.core.env import BRTEnv, EnvState


class GymnasiumEnv(gymnasium.Env):
    """Wrap a BRT environment that uses the ``mjc`` backend as a ``gymnasium.Env``.

    Observations are the environment's observation dictionary. Use ``gymnasium.wrappers.FlattenObservation`` to get a
    flat vector. ``reset(options=...)`` forwards the options to the environment's ``reset`` as task options (e.g.
    ``{"target_position": ...}``). Info values are returned as copies, so changing them never affects the simulation
    or other steps' infos.
    """

    def __init__(self, env: BRTEnv) -> None:
        if env.backend.name != "mjc":
            raise ValueError(
                "GymnasiumEnv requires the 'mjc' backend; use the functional API for MJX/Warp."
            )
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.render_mode = env.config.render_mode
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 1 / env.config.control_timestep,
        }
        self._state: Optional[EnvState] = None

    @property
    def state(self) -> EnvState:
        return self._state

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self._state = self.env.reset(rng=self.np_random, **(options or {}))
        return self._state.observations, _copy_info(self._state.info)

    def step(self, action: np.ndarray):
        self._state = self.env.step(self._state, np.asarray(action))
        s = self._state
        return (
            s.observations,
            float(s.reward),
            bool(s.terminated),
            bool(s.truncated),
            _copy_info(s.info),
        )

    def render(self):
        frames = self.env.render(self._state)
        if frames is None:
            return None
        return frames[0] if len(frames) == 1 else np.concatenate(frames, axis=1)

    def close(self) -> None:
        self.env.close()


def _copy_info(info: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in info.items()}
