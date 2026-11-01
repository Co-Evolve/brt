"""Configure other muscle architectures through the morphology specification and drive them open-loop.

Three tails, each contracted with the same 3 s ramp of all its beams:

* ``dorsal_tip``: a pair of position-controlled dorsal HM beams that only span the three most distal segments (the
  tip bends dorsally, i.e. negative pitch); position control needs a 1 ms timestep (see ``SeahorseTailEnvConfig``).
  Once the tip reaches its pitch limits, the rising beam tension buckles it sideways (~5 deg of roll): as in the
  original model, the dextral and sinistral attachment points differ by up to ~1 mm, and which way and how far the tip
  rolls is sensitive to this asymmetry and to numerical precision.
* ``sinistral``: a single force-controlled ventral-sinistral HM beam over the 6 most distal segments, routed along the
  most lateral attachment points (the tail bends ventrally and to the sinistral side, i.e. negative roll).
* ``mvm_only``: no HM beams, MVM beams that may contract to half their rest length (``mvm_strain=0.5``) on a 7-segment
  tail. As in the original model, both MVM beams of a segment pair attach to the ventral-sinistral plate, so they also
  bend the tail sinistrally.

    python -m biorobot.seahorse.examples.muscle_architectures --backend mjc --render human
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from biorobot.core.env import BRTEnv
from biorobot.seahorse import (
    SeahorseMorphologySpecification,
    default_hm_routing_specification,
    default_seahorse_morphology_specification,
)
from biorobot.seahorse.environment.tail import make_tail_env
from biorobot.seahorse.examples.common import contraction, parse_args, ramp, rollout


def dorsal_tip() -> SeahorseMorphologySpecification:
    specification = default_seahorse_morphology_specification(
        num_segments=11, hm_segment_span=3, p_control=True, mvm_enabled=False
    )
    hm = specification.beam_actuation_specification.hm_beam_actuation_specification
    hm.routing_specifications = [
        default_hm_routing_specification(
            num_segments=11, segment_span=3, sagittal_sides=["dorsal"]
        )
    ]
    return specification


def sinistral() -> SeahorseMorphologySpecification:
    specification = default_seahorse_morphology_specification(
        num_segments=11, hm_segment_span=6, p_control=False, mvm_enabled=False
    )
    hm = specification.beam_actuation_specification.hm_beam_actuation_specification
    hm.f_control_gear = 5.0  # N
    hm.routing_specifications = [
        default_hm_routing_specification(
            num_segments=11,
            segment_span=6,
            coronal_sides=["sinistral"],
            intermediate_points=[0] * 5,  # one per segment between start and end
        )
    ]
    return specification


def mvm_only() -> SeahorseMorphologySpecification:
    return default_seahorse_morphology_specification(
        num_segments=7, hm_segment_span=0, mvm_enabled=True, mvm_strain=0.5
    )


#: Architecture name -> (specification, environment config overrides).
ARCHITECTURES: Dict[str, Tuple[SeahorseMorphologySpecification, dict]] = {
    "dorsal_tip": (
        dorsal_tip(),
        dict(physics_timestep=0.001, num_physics_steps_per_control_step=40),
    ),
    "sinistral": (sinistral(), {}),
    "mvm_only": (mvm_only(), {}),
}


def ramp_all(env: BRTEnv, t: float) -> np.ndarray:
    return contraction(env, ramp(t, 0.0, 3.0))


def main() -> None:
    args = parse_args(__doc__)
    render_mode = None if args.render == "none" else args.render
    for name, (specification, overrides) in ARCHITECTURES.items():
        env = make_tail_env(
            backend=args.backend,
            morphology_specification=specification,
            render_mode=render_mode,
            camera_ids=(0, 2),
            **overrides,
        )
        final = rollout(env, ramp_all)[-1]
        pitch = np.rad2deg(np.sum(final["vertebrae_pitch_joint_pos"]))
        roll = np.rad2deg(np.sum(final["vertebrae_roll_joint_pos"]))
        print(
            f"{name}: {len(env.actuator_names)} beams, total pitch {pitch:.0f} deg, total roll {roll:.0f} deg."
        )
        env.close()


if __name__ == "__main__":
    main()
