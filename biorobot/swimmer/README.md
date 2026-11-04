# The Swimmer Environment

This package contains an abstract anguilliform swimmer: a planar chain of capsules in a viscous fluid, driven by a
position servo on every joint.

Unlike the other BRT animals, the swimmer is not a particular species. It is the simplest body that swims by sending a
wave along itself, which makes it a good first model for asking how body length, taper, joint stiffness and the shape
of the body wave trade speed against energy.

## In-silico swimmer

All quantities are in SI units.

- **Body**: a chain of `num_segments` capsules (radius 15 mm, density 600 kg/m³). The head segment is 0.1 m long, and
  segment lengths change linearly to `taper` × 0.1 m at the tail. The head carries the free joint; the swimmer starts
  straight along +x with its head at the origin, so **head first is -x**.
- **Joints**: a hinge around the vertical axis between every two segments, with a passive torsional spring
  (`stiffness`, default 0.01 N m/rad), damping 0.005 N m s/rad, a ±1.5 rad limit and an armature (rotor inertia) of
  1e-3 kg m².
- **Actuation**: a position servo per joint with kp = 2 N m/rad, whose target is limited to the joint's range. MuJoCo
  clips targets outside it, so rescale your controller's output to the range (`env.action_low`, `env.action_high`)
  rather than relying on clipping; `examples/van_der_pol_cpg.py` shows this for a CPG.
- **Fluid** (`FluidArena`): viscosity 0.1 Pa s (~100 times that of water), density 1000 kg/m³, no gravity and Euler
  integration at 2 ms. MuJoCo's ellipsoid fluid model (`fluidshape="ellipsoid"`) applies drag, added mass and lift to
  every capsule. It does not resolve a flow field, so make no hydrodynamic claims from it; comparisons are internal to
  this model.
- **Contacts**: capsules that are not neighbours collide, which only matters when the body coils. The floor is
  visual only.

The specification (`biorobot/swimmer/specification.py`) is a set of plain dataclasses, one per segment and one per
joint, so you can also give every segment its own radius or every joint its own stiffness.
`default_swimmer_morphology_specification` sets the common parameters:

```python
from biorobot.swimmer import SwimmerMorphology, default_swimmer_morphology_specification

specification = default_swimmer_morphology_specification(
    num_segments=10,  # 0.1 m head segment, so this also sets the body length
    taper=0.6,  # tail segment 0.6 times as long as the head segment
    stiffness=0.003,  # N m/rad
)
print(f"{specification.body_length:.2f} m")  # 0.80 m
SwimmerMorphology(specification).export("swimmer.xml")  # inspect with `python -m mujoco.viewer`
```

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
  the integral of Σ|τ · ω| over the joints. Negative work counts as cost too. It is sampled once per control step, so
  it integrates over every physics step only with one physics step per control step, which is the default.
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
`"mjc"`, check `state.data.warning` after an episode if you search over parameters that can blow up.

**Very thin or light bodies can still become unstable under servo targets that jump every step.** Raise `armature`,
or use `FluidArenaConfiguration(integrator="implicitfast")` (`"mjc"` and `"warp"` only: MJX does not implement it with
fluid forces).

## Examples

```bash
python -m biorobot.swimmer.examples.van_der_pol_cpg --render human
python -m biorobot.swimmer.examples.van_der_pol_cpg --backend warp --render none --num-segments 12
```

The example drives the swimmer with a chain of coupled Van der Pol oscillators, one per segment. With a stronger
forward than backward coupling, the body wave runs from head to tail and the swimmer swims head first, at ~0.15 m/s
for the default 8 segments; swap the couplings and it swims tail first.
