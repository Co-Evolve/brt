import jax.numpy as jnp
import jax.random

from biorobot.brittle_star.environment.directed_locomotion.dual import (
    BrittleStarDirectedLocomotionEnvironment,
)
from biorobot.brittle_star.environment.directed_locomotion.shared import (
    BrittleStarDirectedLocomotionEnvironmentConfiguration,
)
from biorobot.brittle_star.environment.light_escape.dual import BrittleStarLightEscapeEnvironment
from biorobot.brittle_star.environment.light_escape.shared import BrittleStarLightEscapeEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import (
    AquariumArenaConfiguration,
    MJCFAquariumArena,
)
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import (
    default_brittle_star_morphology_specification,
)


def create_morphology() -> MJCFBrittleStarMorphology:
    morphology_spec = default_brittle_star_morphology_specification(
        num_arms=5,
        num_segments_per_arm=6,
        use_p_control=False,
        use_torque_control=True,
        radius_to_strength_factor=400,
    )
    return MJCFBrittleStarMorphology(morphology_spec)


def create_dirlo_env(
        backend: str, render_mode: str
) -> BrittleStarDirectedLocomotionEnvironment:
    morphology = create_morphology()
    arena_config = AquariumArenaConfiguration(attach_target=True)
    arena = MJCFAquariumArena(configuration=arena_config)
    env_config = BrittleStarDirectedLocomotionEnvironmentConfiguration(
        render_mode=render_mode,
        num_physics_steps_per_control_step=10,
        simulation_time=5,
        time_scale=2,
        camera_ids=[0, 1],
        color_contacts=True,
    )
    env = BrittleStarDirectedLocomotionEnvironment.from_morphology_and_arena(
        morphology=morphology, arena=arena, configuration=env_config, backend=backend
    )
    return env


def create_light_escape_env(backend: str, render_mode: str) -> BrittleStarLightEscapeEnvironment:
    morphology = create_morphology()
    arena_config = AquariumArenaConfiguration(sand_ground_color=True)
    arena = MJCFAquariumArena(configuration=arena_config)

    env_config = BrittleStarLightEscapeEnvironmentConfiguration(
        render_mode=render_mode,
        light_perlin_noise_scale=10,
        num_physics_steps_per_control_step=10,
        simulation_time=5,
        time_scale=2,
        camera_ids=[0, 1],
        color_contacts=True,
    )
    env = BrittleStarLightEscapeEnvironment.from_morphology_and_arena(
        morphology=morphology, arena=arena, configuration=env_config, backend=backend
    )
    return env


def clip_and_rescale(
        values: jax.Array,
        original_low: jax.Array,
        original_high: jax.Array,
        new_low: jax.Array,
        new_high: jax.Array,
) -> jax.Array:
    clipped = jnp.clip(values, original_low, original_high)
    # Normalizing to 0-1
    normalized = (clipped - original_low) / (original_high - original_low)

    # Scaling to new range
    scaled = normalized * (new_high - new_low) + new_low

    return scaled


if __name__ == "__main__":
    # Create env
    env = create_dirlo_env("MJX", "human")

    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    env.environment_configuration.render_mode = "human"

    rng = jax.random.PRNGKey(0)
    state = reset_fn(rng)

    i = 0
    while True:
        rng, sub_rng = jax.random.split(rng, 2)
        actions = env.action_space.sample(rng)
        state = step_fn(state=state, action=actions)
        env.render(state=state)

        if state.terminated | state.truncated:
            rng, sub_rng = jax.random.split(rng, 2)
            state = reset_fn(rng)

        i += 1
