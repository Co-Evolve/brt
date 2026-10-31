"""Open-loop "rowing" with a two-armed brittle star: both arms trace an oval, mirrored, with position control."""

from __future__ import annotations

from typing import Any

import biorobot
from biorobot.brittle_star import default_brittle_star_morphology_specification
from biorobot.brittle_star.examples.common import parse_args, render_mode, run
from biorobot.core import EnvState

FREQUENCY = 3.0  # rad/s


def main() -> None:
    args = parse_args(__doc__)
    env = biorobot.make(
        "brittle_star/undirected_locomotion",
        backend=args.backend,
        # Only arms 1 and 4 (the two arms on either side of the +x axis, where the missing arm 0 would be) have segments.
        morphology_specification=default_brittle_star_morphology_specification(
            num_arms=5,
            num_segments_per_arm=[0, 3, 0, 0, 3],
            use_p_control=True,
            radius_to_strength_factor=200,
        ),
        simulation_time=5,
        render_mode=render_mode(args),
        camera_ids=(0, 1),
        color_contacts=True,
    )
    xp = env.xp
    in_plane_range, out_of_plane_range = env.action_high[0], env.action_high[1]

    def policy(state: EnvState, rng: Any):
        # Joints alternate in-plane / out-of-plane along each arm; the in-plane swing is mirrored between the arms.
        time = state.info["time"]
        in_plane = in_plane_range * xp.cos(FREQUENCY * time)
        out_of_plane = 0.4 * out_of_plane_range * xp.sin(FREQUENCY * time)
        is_in_plane = xp.arange(env.mj_model.nu // 2) % 2 == 0
        arm = xp.where(is_in_plane, in_plane, out_of_plane)
        mirrored = xp.where(is_in_plane, -in_plane, out_of_plane)
        return xp.concatenate([arm, mirrored])

    run(env, policy, args)


if __name__ == "__main__":
    main()
