"""Backend-agnostic environment base class.

A BRT environment is a pure ``reset(rng) -> state`` / ``step(state, action) -> state`` interface. Task subclasses
implement a handful of hooks once, using ``self.xp`` (``numpy`` or ``jax.numpy``) and ``self.backend``; the same task
then runs on

* ``backend="mjc"``: regular MuJoCo on the CPU. ``state.data`` is a ``mujoco.MjData`` that is updated **in place**.
* ``backend="mjx"`` / ``backend="warp"``: MuJoCo XLA with its JAX or MuJoCo Warp implementation. ``state`` is a JAX
  pytree, so ``jax.jit(env.step)`` and ``jax.vmap(env.reset)(keys)`` work as expected.

Because ``mjc`` updates ``state.data`` in place, hooks that need values from the previous timestep (e.g. rewards based
on progress) must read them from ``previous_state.observations`` or ``previous_state.info``, which are fresh
dictionaries every step, and never from ``previous_state.data``.
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

import mujoco
import numpy as np

from biorobot.core.backend import Array, Backend, make_backend
from biorobot.core.builder import Arena, Morphology, compose


@dataclass
class EnvConfig:
    """Settings shared by every BRT environment.

    Args:
        simulation_time: Episode length in seconds; the episode is truncated afterwards.
        num_physics_steps_per_control_step: Physics steps per ``env.step`` call (action repeat).
        physics_timestep: Physics timestep in seconds.
        solver_iterations: Override the constraint solver's iteration count (lower is faster, less accurate).
        solver_ls_iterations: Override the solver's line-search iteration count.
        disable_eulerdamp: Disable implicit damping in the Euler integrator (recommended with MJX).
        render_mode: ``None``, ``"rgb_array"`` (``render`` returns frames) or ``"human"`` (frames are shown in an
            OpenCV window).
        render_size: ``(height, width)`` of rendered frames.
        camera_ids: Cameras to render; frames of multiple cameras are placed side by side in ``"human"`` mode.
        num_envs: Maximum number of environments you will vmap together. Only ``backend="warp"`` needs it: MuJoCo
            Warp preallocates a single contact buffer for the whole batch.
    """

    simulation_time: float = 10.0
    num_physics_steps_per_control_step: int = 10
    physics_timestep: float = 0.002
    solver_iterations: Optional[int] = None
    solver_ls_iterations: Optional[int] = None
    disable_eulerdamp: bool = False
    render_mode: Optional[str] = None
    render_size: Tuple[int, int] = (480, 640)
    camera_ids: Sequence[int] = (0,)
    num_envs: int = 1

    @property
    def control_timestep(self) -> float:
        return self.num_physics_steps_per_control_step * self.physics_timestep

    @property
    def total_num_control_steps(self) -> int:
        return int(np.ceil(self.simulation_time / self.control_timestep))


ConfigT = TypeVar("ConfigT", bound=EnvConfig)


def make_config(
    config_class: Type[ConfigT], config: Optional[ConfigT] = None, **overrides: Any
) -> ConfigT:
    """A copy of ``config`` (default: ``config_class()``) with ``overrides`` applied; the caller's object is untouched.

    Raises ``TypeError`` for a config of another class or an unknown override.
    """
    if config is None:
        config = config_class()
    elif not isinstance(config, config_class):
        raise TypeError(
            f"Expected a {config_class.__name__}, got a {type(config).__name__}."
        )
    field_names = {f.name for f in dataclasses.fields(config_class)}
    unknown = sorted(set(overrides) - field_names)
    if unknown:
        raise TypeError(f"{config_class.__name__} has no option(s) {unknown}.")
    return dataclasses.replace(config, **overrides)


@dataclass(frozen=True)
class Observable:
    """A named observation: ``fn(data, info)`` returns a 1D array within ``[low, high]``."""

    name: str
    low: np.ndarray
    high: np.ndarray
    fn: Callable[[Any, Dict[str, Array]], Array]

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(np.shape(self.low))


@dataclass
class EnvState:
    data: Any
    observations: Dict[str, Array]
    reward: Array
    terminated: Array
    truncated: Array
    info: Dict[str, Array]
    rng: Any

    def replace(self, **changes: Any) -> "EnvState":
        return dataclasses.replace(self, **changes)


try:  # Make EnvState a JAX pytree so MJX states can be jitted/vmapped.
    import jax

    jax.tree_util.register_dataclass(
        EnvState,
        data_fields=[f.name for f in dataclasses.fields(EnvState)],
        meta_fields=[],
    )
except ImportError:  # pragma: no cover
    pass  # jax is a hard dependency, but keep the core importable without it.


class BRTEnv(abc.ABC):
    """Base class of every BRT environment. See the module docstring for the backend contract."""

    #: Name of the morphology this environment expects (used as the prefix of all morphology element names).
    morphology_name: str = ""

    def __init__(
        self,
        spec: mujoco.MjSpec,
        config: EnvConfig,
        backend: str = "mjc",
    ) -> None:
        self.config = config
        self._configure_options(spec)
        self.mj_model = spec.compile()
        self.backend: Backend = make_backend(
            backend, self.mj_model, num_envs=config.num_envs
        )
        self.xp = self.backend.xp

        self._cache_references(self.mj_model)
        self.observables: List[Observable] = self._create_observables()

        # One render model and renderer for all (batched) environments: _prepare_render_model adapts the model to each
        # environment just before it is rendered.
        self._render_model: Optional[mujoco.MjModel] = None
        self._renderer: Optional[mujoco.Renderer] = None

    # ------------------------------------------------------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: Morphology,
        arena: Arena,
        config: EnvConfig,
        backend: str = "mjc",
    ) -> "BRTEnv":
        if cls.morphology_name and morphology.name != cls.morphology_name:
            raise ValueError(
                f"{cls.__name__} expects a '{cls.morphology_name}' morphology, got '{morphology.name}'."
            )
        cls._validate_arena(arena)
        spec = compose(arena=arena, morphology=morphology, free_joint=True)
        return cls(spec=spec, config=config, backend=backend)

    @classmethod
    def _validate_arena(cls, arena: Arena) -> None:
        """Raise if ``arena`` cannot be used with this environment."""

    def _configure_options(self, spec: mujoco.MjSpec) -> None:
        spec.option.timestep = self.config.physics_timestep
        if self.config.solver_iterations is not None:
            spec.option.iterations = self.config.solver_iterations
        if self.config.solver_ls_iterations is not None:
            spec.option.ls_iterations = self.config.solver_ls_iterations
        if self.config.disable_eulerdamp:
            spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP

    def _cache_references(self, mj_model: mujoco.MjModel) -> None:
        """Hook to look up ids/addresses once (e.g. body ids, qpos addresses) before observables are created."""

    # ------------------------------------------------------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def action_low(self) -> np.ndarray:
        return self.mj_model.actuator_ctrlrange[:, 0].copy()

    @property
    def action_high(self) -> np.ndarray:
        return self.mj_model.actuator_ctrlrange[:, 1].copy()

    @property
    def actuator_names(self) -> List[str]:
        return [self.mj_model.actuator(i).name for i in range(self.mj_model.nu)]

    @property
    def action_space(self):
        import gymnasium

        return gymnasium.spaces.Box(
            low=self.action_low, high=self.action_high, dtype=np.float64
        )

    @property
    def observation_space(self):
        import gymnasium

        return gymnasium.spaces.Dict(
            {
                obs.name: gymnasium.spaces.Box(
                    low=obs.low, high=obs.high, shape=obs.shape, dtype=np.float64
                )
                for obs in self.observables
            }
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------------------------------------------------------
    def reset(self, rng: Any, **task_options: Any) -> EnvState:
        """Start a new episode.

        Args:
            rng: A seed or ``np.random.Generator`` (``mjc``), or a JAX PRNG key (``mjx``/``warp``).
            **task_options: Task-specific options, e.g. ``target_position`` (see the environment's ``_reset``).
        """
        rng = self.backend.make_rng(rng)
        data = self.backend.make_data()
        data, info, rng = self._reset(data=data, rng=rng, **task_options)
        data = self.backend.forward(data)

        xp = self.xp
        state = EnvState(
            data=data,
            observations={},
            reward=xp.asarray(0.0),
            terminated=xp.asarray(False),
            truncated=xp.asarray(False),
            info=info,
            rng=rng,
        )
        state = state.replace(observations=self._get_observations(state))
        state = state.replace(info=self._get_info(state=state, previous_state=None))
        return state

    def step(self, state: EnvState, action: Array) -> EnvState:
        data = self.backend.step(
            state.data, action, self.config.num_physics_steps_per_control_step
        )
        next_state = state.replace(data=data)
        next_state = next_state.replace(observations=self._get_observations(next_state))
        next_state = next_state.replace(
            info=self._get_info(state=next_state, previous_state=state)
        )
        return next_state.replace(
            reward=self.xp.asarray(
                self._get_reward(state=next_state, previous_state=state)
            ),
            terminated=self.xp.asarray(self._get_terminated(state=next_state)),
            truncated=self.xp.asarray(self._get_truncated(state=next_state)),
        )

    def _get_observations(self, state: EnvState) -> Dict[str, Array]:
        # xp.array copies: with "mjc", an observable that slices the data would otherwise be a view that changes with
        # the next (in-place) step.
        return {
            obs.name: self.xp.reshape(
                self.xp.array(obs.fn(state.data, state.info), dtype=float), (-1,)
            )
            for obs in self.observables
        }

    # ------------------------------------------------------------------------------------------------------------------
    # Task hooks
    # ------------------------------------------------------------------------------------------------------------------
    @abc.abstractmethod
    def _create_observables(self) -> List[Observable]: ...

    @abc.abstractmethod
    def _reset(
        self, data: Any, rng: Any, **task_options: Any
    ) -> Tuple[Any, Dict[str, Array], Any]:
        """Set the initial configuration. Returns ``(data, info, rng)``; ``info`` holds per-episode task variables.

        Everything that may differ between (vmapped) episodes must live in ``data`` (e.g. ``qpos`` or ``mocap_pos``
        for targets) or ``info``, never in the model.
        """

    def _get_info(
        self, state: EnvState, previous_state: Optional[EnvState]
    ) -> Dict[str, Array]:
        """Return a *new* info dict. By default, per-episode variables are carried over and ``time`` is updated."""
        return {**state.info, "time": state.data.time}

    @abc.abstractmethod
    def _get_reward(self, state: EnvState, previous_state: EnvState) -> Array: ...

    def _get_terminated(self, state: EnvState) -> Array:
        return False

    def _get_truncated(self, state: EnvState) -> Array:
        # Truncate after exactly total_num_control_steps steps. data.time accumulates round-off (differently in float64
        # and float32), so comparing it with simulation_time itself can take one step more or less per backend.
        config = self.config
        return (
            state.data.time
            >= (config.total_num_control_steps - 0.5) * config.control_timestep
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------------------------------------------------------
    def _prepare_render_model(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        observations: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
    ) -> List[int]:
        """Hook to modify the render model for visualisation of one environment, e.g. recolour geoms.

        A single render model is shared by all environments of a batched state and is passed to this hook before each
        of them is rendered, so set every property you change (do not rely on what the previous call left behind).
        Returns the ids of textures that changed and must be re-uploaded to the GPU.
        """
        return []

    def render(self, state: EnvState) -> Optional[List[np.ndarray]]:
        """Render all cameras in ``config.camera_ids`` for every environment in ``state`` (batched MJX states too).

        Returns the frames (ordered environment-major, camera-minor) in ``"rgb_array"`` mode, or shows them and
        returns ``None`` in ``"human"`` mode.
        """
        mode = self.config.render_mode
        if mode not in ("rgb_array", "human"):
            raise ValueError(
                "Set EnvConfig.render_mode to 'rgb_array' or 'human' to render."
            )

        batched = np.ndim(state.data.time) > 0
        num_envs = int(np.shape(state.data.time)[0]) if batched else 1

        frames = []
        for env_index in range(num_envs):
            index = env_index if batched else None
            mj_data = self.backend.to_mj_data(state.data, index=index)
            observations = _take(state.observations, index)
            info = _take(state.info, index)

            if self._renderer is None:
                height, width = self.config.render_size
                self._render_model = _copy_model(self.mj_model)
                self._renderer = mujoco.Renderer(
                    self._render_model, height=height, width=width
                )
            model, renderer = self._render_model, self._renderer
            textures = self._prepare_render_model(model, mj_data, observations, info)
            if textures:
                renderer._gl_context.make_current()
                for texture_id in textures:
                    mujoco.mjr_uploadTexture(model, renderer._mjr_context, texture_id)

            for camera_id in self.config.camera_ids:
                renderer.update_scene(mj_data, camera=camera_id)
                frames.append(renderer.render().copy())

        if mode == "human":
            show_frames(frames, num_cameras=len(self.config.camera_ids))
            return None
        return frames

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None
        self._render_model = None
        if self.config.render_mode == "human":
            import cv2

            cv2.destroyAllWindows()


def _copy_model(model: mujoco.MjModel) -> mujoco.MjModel:
    import copy

    return copy.deepcopy(model)


def _take(tree: Dict[str, Array], index: Optional[int]) -> Dict[str, np.ndarray]:
    def take(value):
        value = np.asarray(value)
        if index is not None and value.ndim > 0:
            return value[index]
        return value

    return {key: take(value) for key, value in tree.items()}


def tile_frames(frames: List[np.ndarray], num_cameras: int) -> np.ndarray:
    """Place the cameras of one environment side by side and stack environments vertically."""
    rows = [
        np.concatenate(frames[i : i + num_cameras], axis=1)
        for i in range(0, len(frames), num_cameras)
    ]
    return np.concatenate(rows, axis=0)


def show_frames(frames: List[np.ndarray], num_cameras: int) -> None:
    import cv2

    image = tile_frames(frames, num_cameras=num_cameras)
    cv2.imshow("BRT", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
