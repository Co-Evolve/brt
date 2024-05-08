from typing import Tuple, List

import chex
import cv2
import jax.numpy as jnp
import jax.random
import numpy as np

from biorobot.brittle_star.environment.directed_locomotion.dual import (
    BrittleStarDirectedLocomotionEnvironment,
)
from biorobot.brittle_star.environment.directed_locomotion.shared import (
    BrittleStarDirectedLocomotionEnvironmentConfiguration,
)
from biorobot.brittle_star.mjcf.arena.aquarium import (
    AquariumArenaConfiguration,
    MJCFAquariumArena,
)
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import (
    default_brittle_star_morphology_specification,
)


def post_render(
    render_output: List[np.ndarray],
    environment_configuration: BrittleStarDirectedLocomotionEnvironmentConfiguration,
) -> None:
    if environment_configuration.render_mode == "human":
        return

    num_cameras = len(environment_configuration.camera_ids)
    num_envs = len(render_output) // num_cameras

    if num_cameras > 1:
        # Horizontally stack frames of the same environment
        frames_per_env = np.array_split(render_output, num_envs)
        render_output = [
            np.concatenate(env_frames, axis=1) for env_frames in frames_per_env
        ]

    # Vertically stack frames of different environments
    stacked_frames = np.concatenate(render_output, axis=0)
    cv2.imshow("render", stacked_frames)
    cv2.waitKey(1)


def create_env(
    backend: str, render_mode: str
) -> BrittleStarDirectedLocomotionEnvironment:
    morphology_spec = default_brittle_star_morphology_specification(
        num_arms=5, num_segments_per_arm=5, use_p_control=True
    )
    morphology = MJCFBrittleStarMorphology(morphology_spec)
    arena_config = AquariumArenaConfiguration(attach_target=True)
    arena = MJCFAquariumArena(configuration=arena_config)

    env_config = BrittleStarDirectedLocomotionEnvironmentConfiguration(
        render_mode=render_mode,
        num_physics_steps_per_control_step=10,
        simulation_time=5,
        time_scale=1,
        camera_ids=[0, 1],
        color_contacts=True,
    )
    env = BrittleStarDirectedLocomotionEnvironment.from_morphology_and_arena(
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

    state = reset_fn(env_rng, [-1.0, 0.0, 0.05])
    while True:
        action, action_rng = action_sample_fn(action_rng)
        state = step_fn(state=state, action=action)
        print(state.observations["joint_actuator_force"])
        print(state.observations["actuator_force"])
        print()
        post_render(env.render(state=state), env.environment_configuration)
        if state.terminated | state.truncated:
            state = reset_fn(env_rng)
    env.close()
