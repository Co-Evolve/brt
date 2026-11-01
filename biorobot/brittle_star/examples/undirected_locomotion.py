"""Undirected locomotion with an open-loop oscillator: arm 0 is lifted, the other arms swing back and forth."""

from __future__ import annotations

from typing import Any

import biorobot
from biorobot.brittle_star import default_brittle_star_morphology_specification
from biorobot.brittle_star.examples.common import parse_args, render_mode, run
from biorobot.core import EnvState

NUM_ARMS = 5


def main() -> None:
    args = parse_args(__doc__)
    env = biorobot.make(
        "brittle_star/undirected_locomotion",
        backend=args.backend,
        morphology_specification=default_brittle_star_morphology_specification(
            num_arms=NUM_ARMS,
            num_segments_per_arm=6,
            use_p_control=True,
            radius_to_strength_factor=200,
        ),
        simulation_time=5,
        render_mode=render_mode(args),
        camera_ids=(0, 1),
        color_contacts=True,
    )
    xp = env.xp

    def policy(state: EnvState, rng: Any):
        # Position targets per arm; joints alternate in-plane / out-of-plane along each arm. Normalised targets in
        # [-1, 1] are scaled to the joint ranges.
        time = state.info["time"]
        num_joints = env.mj_model.nu // NUM_ARMS
        in_plane = xp.arange(num_joints) % 2 == 0
        swing = xp.where(in_plane, xp.cos(5 * time), xp.sin(5 * time))
        direction = xp.array([0.0, -1.0, -1.0, 1.0, 1.0])[:, None]
        targets = xp.where(in_plane[None], direction * swing, swing)
        targets = xp.where(
            (xp.arange(NUM_ARMS) == 0)[:, None], ~in_plane[None] * 1.0, targets
        )
        return targets.reshape(-1) * env.action_high

    run(env, policy, args)


if __name__ == "__main__":
    main()
