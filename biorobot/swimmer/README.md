# The Swimmer Environment

This package contains an abstract anguilliform swimmer: a planar chain of capsules in a viscous fluid, driven by a
position servo on every joint. It is the toy swimmer of the review on physics-based simulation of biological locomotion
(*Physics of Life Reviews*). The review uses it to show how the choice of behavioural descriptors shapes what a
quality-diversity search (MAP-Elites) finds.

Unlike the other BRT animals, the swimmer is not a particular species. It is the simplest body that swims by sending a
wave along itself, which makes it a good first model for asking how body length, taper, joint stiffness and the shape
of the body wave trade speed against energy.

## In-silico swimmer

All quantities are in SI units. The defaults are the review's.

- **Body**: a chain of `num_segments` capsules (radius 15 mm, density 600 kg/m³). The head segment is 0.1 m long, and
  segment lengths change linearly to `taper` × 0.1 m at the tail. The head carries the free joint; the swimmer starts
  straight along +x with its head at the origin, so **head first is -x**.
- **Joints**: a hinge around the vertical axis between every two segments, with a passive torsional spring
  (`stiffness`, default 0.01 N m/rad), damping 0.005 N m s/rad and a ±1.5 rad limit.
- **Actuation**: a position servo per joint with kp = 2 N m/rad. Its targets range over ±2π rad, far beyond the
  joint limits, on purpose: the review's CPGs command targets up to ~4.9 rad, so the servos act as soft torque sources
  (τ = 2 · (target − angle)) and clipping their targets to the joint range would change the gait.
- **Fluid** (`FluidArena`): viscosity 0.1 Pa s (~100 times that of water), density 1000 kg/m³, no gravity and Euler
  integration at 2 ms. MuJoCo's ellipsoid fluid model (`fluidshape="ellipsoid"`) applies drag, added mass and lift to
  every capsule. It does not resolve a flow field, so make no hydrodynamic claims from it; comparisons are internal to
  this model.
- **Contacts**: capsules that are not neighbours collide, which only matters when the body coils. The floor is
  visual only.

The specification (`biorobot/swimmer/specification.py`) is a set of plain dataclasses, one per segment and one per
joint, so you can also give every segment its own radius or every joint its own stiffness.
`default_swimmer_morphology_specification` covers the review's search space:

```python
from biorobot.swimmer import SwimmerMorphology, default_swimmer_morphology_specification

specification = default_swimmer_morphology_specification(
    num_segments=10,  # the review searches over 4, 6, 8, 10 and 12
    taper=0.6,  # tail segment 0.6 times as long as the head segment
    stiffness=0.003,  # N m/rad
)
print(f"{specification.body_length:.2f} m")  # 0.80 m
SwimmerMorphology(specification).export("swimmer.xml")  # inspect with `python -m mujoco.viewer`
```

### Reproducing the review's runs

**The review's runs used a joint limit of ±1.5 degrees, not ±1.5 rad.** Their MJCF wrote `range="-1.5 1.5"` without
`<compiler angle="radian"/>`, so MuJoCo read it in degrees. The servos push the joints far past that limit anyway
(joint angles still reach ~1.8 rad), so the bending is resisted mainly by the soft joint-limit constraint rather than by
the springs. BRT defaults to the documented ±1.5 rad. To reproduce the review's model, pass
`joint_range=REVIEW_JOINT_RANGE`:

```python
from biorobot.swimmer import REVIEW_JOINT_RANGE, default_swimmer_morphology_specification

specification = default_swimmer_morphology_specification(
    num_segments=8, taper=1.0, stiffness=0.01, joint_range=REVIEW_JOINT_RANGE
)
```

With it, the swimmer simulates exactly like the original MJCF model (`test_matches_original_model` in
[`tests/test_swimmer.py`](https://github.com/Co-Evolve/brt/blob/main/tests/test_swimmer.py) checks this to 1e-10 m).
Two things still differ from the review's numbers:

- The original model rounded segment lengths to 0.1 mm and stiffnesses to 1e-6 N m/rad when writing its XML; BRT does
  not round. Round them yourself if you need the same last digits.
- BRT needs MuJoCo 3.17 or newer, while the review's runs used MuJoCo 3.8.1. Energy and joint excursion barely change
  between the two (< 0.1% on a dozen random swimmers), but the speed of slow swimmers can (up to ~0.03 m/s), so
  rerun the searches rather than comparing against the old grids.

The CPG, the MAP-Elites searches and the descriptors of the review are not part of BRT; they belong to the experiment.
[`examples/van_der_pol_cpg.py`](https://github.com/Co-Evolve/brt/blob/main/biorobot/swimmer/examples/van_der_pol_cpg.py)
shows the CPG driving this environment.

## Environment

```python
import numpy as np

import biorobot

env = biorobot.make("swimmer/head_first_swimming", backend="mjc")  # or "warp"
state = env.reset(0)
phase = 2 * np.pi * np.arange(env.mj_model.nu) / env.mj_model.nu
while not state.truncated:
    t = float(state.info["time"])
    state = env.step(state, 0.6 * np.sin(2 * np.pi * 1.5 * t - phase))  # head-to-tail body wave
print(f"{state.info['head_first_distance']:+.2f} m, {state.info['actuator_work']:.1f} J")
```

### Head-first swimming

- **Actions**: one servo target (rad) per joint, head to tail.
- **Reward**: the head's displacement (m) along -x during the step. Tail-first swimming gets a negative reward, so a
  body wave that runs the wrong way is penalised, not ignored.
- **Info**: `head_first_distance`, the head's total displacement (m) along -x since the reset, and `actuator_work` (J),
  the integral of Σ|τ · ω| over the joints. Negative work counts as cost too. It is read once per control step, so it
  is the review's energy only with one physics step per control step, which is the default.
- **Observations**: `joint_position`, `joint_velocity`, `joint_actuator_force` (servo torques), and the head's
  `head_position`, `head_rotation` (extrinsic xyz Euler angles), `head_linear_velocity` and `head_angular_velocity`.
- **Episode**: 10 s, a new action every 2 ms physics step (what a CPG integrated alongside the simulation needs).
  Raise `num_physics_steps_per_control_step` for RL. Never terminates.

### Backends

| Backend | Supported | Why |
|---|---|---|
| `"mjc"` | yes | |
| `"warp"` | yes | matches MuJoCo C (`test_warp_matches_mjc`) |
| `"mjx"` | **no**, raises `NotImplementedError` | MJX's JAX implementation ignores `fluidshape="ellipsoid"` and applies the inertia-based fluid model instead, without a warning (~13% different fluid forces for this body). BRT refuses such models rather than letting you simulate different physics. |

If you need `"mjx"`, build the swimmer with `fluid_model="inertia"`, MuJoCo's default fluid model, which MJX does
support. That is a *different* fluid model (forces from each body's equivalent inertia box, not from per-capsule
drag, added mass and lift), so results are not comparable with the ellipsoid model's.

**MuJoCo C silently resets the simulation after an instability** (bad qpos, qvel or qacc) and keeps going. With
`"mjc"`, check `state.data.warning` after an episode, as the review did, if you search over parameters that can blow
up (e.g. very coarse timesteps).

## Examples

```bash
python -m biorobot.swimmer.examples.van_der_pol_cpg --render human
python -m biorobot.swimmer.examples.van_der_pol_cpg --backend warp --render none --num-segments 12
```

The example drives the swimmer with a chain of coupled Van der Pol oscillators, the review's CPG. With a stronger
forward than backward coupling, the body wave runs from head to tail and the swimmer swims head first, at ~0.15 m/s
for the default 8 segments; swap the couplings and it swims tail first.
