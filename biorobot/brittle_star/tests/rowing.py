from typing import Dict

import chex
import jax.random
import numpy as np
from gymnasium.envs.tabular.blackjack import EnvState
from moojoco.environment.mjx_spaces import Box

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
    NUM_SEGMENTS_PER_ARM = 3
    morphology_spec = default_brittle_star_morphology_specification(
        num_arms=5,
        num_segments_per_arm=[0, NUM_SEGMENTS_PER_ARM, 0, 0, NUM_SEGMENTS_PER_ARM],
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


def generate_rowing_action(
    action_space: Box, observation_space: Dict, state: EnvState
) -> chex.Array:
    """
    Use P-control to generate oval motion with the bases of the arms (up to half of segments)
    Set force of remaining segments to zero
    """

    NUM_ACTUATED_SEGMENTS = 3
    NUM_ARMS = 2

    time = state.info["time"]

    try:
        ip_rom = observation_space.spaces["joint_position"].high[0]
        oop_rom = observation_space.spaces["joint_position"].high[1]
    except Exception:
        ip_rom = observation_space["joint_position"].high[0]
        oop_rom = observation_space["joint_position"].high[1]

    frequency = 3
    ip_pos = ip_rom * np.cos(frequency * time)
    oop_pos = 0.4 * oop_rom * np.sin(frequency * time)

    positions_per_arm = np.zeros_like(action_space.low).reshape((NUM_ARMS, -1))
    positions_per_arm[0, ::2] = ip_pos
    positions_per_arm[1, ::2] = -ip_pos
    positions_per_arm[:, 1::2] = oop_pos

    return positions_per_arm.flatten()

    target_positions = positions_per_arm.flatten()
    current_positions = state.observations["joint_position"]

    forces = 10 * (target_positions - current_positions)

    # Only actuate targeted segments
    forces_per_arm = forces.reshape((NUM_ARMS, -1))
    forces_per_arm[:, NUM_ACTUATED_SEGMENTS * 2 :] = 0
    forces = forces_per_arm.flatten()

    return forces

    # Only actuate targeted segments
    # forces_per_arm[:, NUM_ACTUATED_SEGMENTS * 2:] = 0
    #
    # forces = forces_per_arm.flatten()
    #
    # Rescale
    # forces = forces * (action_space.high - action_space.low) - action_space.low

    return positions_per_arm.flatten()


if __name__ == "__main__":
    BACKEND = "MJC"
    RENDER_MODE = "human"

    env = create_env(backend=BACKEND, render_mode=RENDER_MODE)

    if BACKEND == "MJC":
        env_rng = np.random.RandomState(0)
        step_fn = env.step
        reset_fn = env.reset
    else:
        env_rng = jax.random.PRNGKey(0)

        step_fn = jax.jit(env.step)
        reset_fn = jax.jit(env.reset)

    state = reset_fn(env_rng)
    while True:
        action = generate_rowing_action(env.action_space, env.observation_space, state)
        state = step_fn(state=state, action=action)
        env.render(state=state)
        if state.truncated or state.terminated:
            state = reset_fn(env_rng)

    env.close()
