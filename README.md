# Bio-Inspired Robotics Testbed (BRT)

The **Bio-Inspired Robotics Testbed (BRT)** is a collection of MuJoCo simulation environments of animals. It serves
two communities:

- **Roboticists** get benchmarks for the *design and control* of bio-inspired robots: parameterised morphologies,
  tasks with rewards, and fast, vectorised simulation on CPU or GPU.
- **Biologists** get robots as a *controlled comparative method*: change one trait of an in-silico animal (arm count,
  leg strength, a dragline, the span of a tail muscle, ...) while keeping everything else fixed, and study its effect
  on behaviour, ecology and evolution.

**Live demos** of every animal run in your browser at **https://co-evolve.github.io/brt/**.

## Environments

| Animal | Tasks (`biorobot.make` name) | Docs |
|---|---|---|
| Brittle star | `brittle_star/undirected_locomotion`, `brittle_star/directed_locomotion`, `brittle_star/light_escape` | [README](https://github.com/Co-Evolve/brt/blob/main/biorobot/brittle_star/README.md) |
| Jumping spider | `jumping_spider/long_jump`, `jumping_spider/platform_jump`, `jumping_spider/directed_jump` | [README](https://github.com/Co-Evolve/brt/blob/main/biorobot/jumping_spider/README.md) |
| Seahorse tail | `seahorse/tail` (no task: a muscle-architecture playground for open-loop control) | [README](https://github.com/Co-Evolve/brt/blob/main/biorobot/seahorse/README.md) |
| Swimmer | `swimmer/head_first_swimming` (`"mjc"` and `"warp"` only, see below) | [README](https://github.com/Co-Evolve/brt/blob/main/biorobot/swimmer/README.md) |

Each animal's README covers its biology, the in-silico morphology, the observations, rewards and configuration of
each task, and runnable examples. The seahorse tail (Marzougui et al., [2025](https://doi.org/10.1098/rsif.2024.0876))
has no task: its reward is always 0, and you study muscle architectures by configuring its morphology and driving it
open-loop. To get a feel for it first, try the
[seahorse muscle playground](https://co-evolve.github.io/seahorse-muscle-architecture/).

## Installation

```bash
pip install biorobot              # MuJoCo (CPU) and MJX (JAX: CPU, GPU, TPU)
pip install "biorobot[cuda]"      # + JAX with CUDA 12, to run MJX on NVIDIA GPUs
pip install "biorobot[warp]"      # + NVIDIA Warp, for the "warp" backend
```

BRT needs Python 3.11 or newer. To work on BRT itself: `pip install -e ".[dev]"` (adds pytest and black).

## Quickstart

```python
import numpy as np

import biorobot

print(biorobot.available())  # all registered environments

env = biorobot.make("brittle_star/undirected_locomotion", backend="mjc", simulation_time=5)
rng = np.random.default_rng(0)

state = env.reset(rng)
while not (state.terminated or state.truncated):
    action = rng.uniform(env.action_low, env.action_high)
    state = env.step(state, action)
print(f"t = {state.info['time']:.2f} s, last reward = {state.reward:.4f}")
```

An environment is a pure `reset(rng) -> state` / `step(state, action) -> state` interface. The state holds the MuJoCo
data, a dictionary of `observations`, the `reward`, the `terminated` and `truncated` flags and task `info`. Keyword
arguments of `biorobot.make` override fields of the task's configuration dataclass (`simulation_time`,
`render_mode`, `camera_ids`, ...) or replace its `morphology_specification` / `arena_configuration`.

Set `render_mode="human"` to watch in an OpenCV window, or `render_mode="rgb_array"` to get RGB frames from
`env.render(state)` and save them with `biorobot.utils.save_video(frames, "episode.mp4", fps=1 / env.config.control_timestep)`.

## Backends

The same task implementation runs on three simulation backends; pick one with `backend=...`:

| Backend | Simulator | Use it for |
|---|---|---|
| `"mjc"` | MuJoCo (C), numpy, CPU | Debugging, visualisation, single environments, evolutionary/morphology search on many CPU cores, Gymnasium-based RL libraries. |
| `"mjx"` | MJX with its JAX implementation | Thousands of parallel environments on a GPU/TPU with `jax.jit` and `jax.vmap`; end-to-end JAX training loops. |
| `"warp"` | MJX with the MuJoCo Warp implementation | Same JAX interface as `"mjx"`, but faster on NVIDIA GPUs, especially for contact-rich models. Needs `biorobot[warp]`; pass `num_envs` (see below). |

**A backend that would silently simulate a model wrongly refuses it.** Creating the environment then raises
`NotImplementedError` and names the backends that do work. Today this only concerns the ellipsoid fluid model
(`fluidshape="ellipsoid"`, used by the swimmer), which MJX's JAX implementation replaces by the inertia-based fluid model
without a warning; MuJoCo Warp supports it. `biorobot.core.unsupported_features(mj_model, backend)` lists what a
backend lacks for a model.

With `"mjx"` and `"warp"`, the state is a JAX pytree, so `reset` and `step` can be jitted and vmapped. MuJoCo Warp
allocates a single contact buffer for all vmapped environments, so with `"warp"` pass the largest batch size you will
use as `num_envs` when creating the environment (it is ignored by the other backends):

```python
import jax

import biorobot

num_envs = 16
env = biorobot.make("jumping_spider/long_jump", backend="mjx", num_envs=num_envs)  # or "warp"
reset = jax.jit(jax.vmap(env.reset))
step = jax.jit(jax.vmap(env.step))

state = reset(jax.random.split(jax.random.PRNGKey(0), num_envs))
actions = jax.random.uniform(
    jax.random.PRNGKey(1), (num_envs, env.mj_model.nu), minval=env.action_low, maxval=env.action_high
)
state = step(state, actions)
print(state.reward.shape)  # (16,)
```

A batched state can be rendered too: `env.render(state)` returns the frames of every environment (rendered one after
the other with a single renderer).

## Gymnasium

`GymnasiumEnv` wraps an `"mjc"` environment as a `gymnasium.Env` for standard RL libraries. Observations are a
dictionary; wrap with `gymnasium.wrappers.FlattenObservation` for a flat vector.

```python
import gymnasium

import biorobot
from biorobot.core.gym import GymnasiumEnv

env = gymnasium.wrappers.FlattenObservation(GymnasiumEnv(biorobot.make("brittle_star/light_escape")))
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Design

- **Models are built in Python with `mujoco.MjSpec`.** Every morphology is generated from a *specification*: plain,
  nested Python dataclasses (e.g. number of arms and segments, joint ranges, actuator strength). Change a field and you
  get a new, valid robot, which is what morphology optimisation and comparative studies need. Arenas are built the
  same way from configuration dataclasses. Both expose the raw `MjSpec` as `.spec`, so you can still edit anything by
  hand before the environment compiles it, or `export()` it to MJCF XML.
- **One task implementation, three backends.** Tasks are written once against `env.xp` (`numpy` or `jax.numpy`) and
  a small backend interface (`biorobot/core/backend.py`). Everything that varies per episode (initial poses, targets,
  light maps) lives in the simulation data or the state's `info`, never in the model, so vmapped environments can
  share one model.
- **A registry** (`biorobot.make`, `biorobot.available`) builds each task with a sensible default morphology and arena.

For example, a brittle star with three arms of eight segments and position control:

```python
import biorobot
from biorobot.brittle_star import default_brittle_star_morphology_specification

specification = default_brittle_star_morphology_specification(
    num_arms=3, num_segments_per_arm=8, use_p_control=True
)
env = biorobot.make("brittle_star/directed_locomotion", morphology_specification=specification)
print(env.mj_model.nu)  # 48 actuators: 3 arms x 8 segments x 2 joints
```

## Adding a new animal

1. Read the core in [`biorobot/core`](https://github.com/Co-Evolve/brt/tree/main/biorobot/core): `builder.py`
   (`Morphology`, `Arena`, `compose`), `env.py` (`BRTEnv`, `EnvConfig` and the task hooks), `observables.py` and
   `backend.py`. [`tests/test_core.py`](https://github.com/Co-Evolve/brt/blob/main/tests/test_core.py) contains a
   complete toy environment in about 60 lines; start from it.
2. Create `biorobot/<animal>/` with `specification.py` (dataclasses and a `default_..._specification`),
   `morphology.py` and `arena.py` (`MjSpec` builders), and `environment/` with one module per task that registers a
   factory with `@biorobot.register("<animal>/<task>")`. Add the package to `_ANIMAL_PACKAGES` in
   `biorobot/registry.py`. The brittle star and jumping spider packages are complete examples. Put mesh or other
   data files under `biorobot/<animal>/assets/` and add them to `[tool.setuptools.package-data]` in `pyproject.toml`
   and to `MANIFEST.in` (like the seahorse's meshes), otherwise they are missing from the wheel.
3. Two rules keep a task correct on every backend: write it with `self.xp` and without Python branching on simulated
   values (use `xp.where`), and compute rewards from `previous_state.observations` or `previous_state.info`, never from
   `previous_state.data` (with `"mjc"` the data is updated in place).
   If your model needs a feature that a backend gets silently wrong, add it to `unsupported_features` in
   `biorobot/core/backend.py`, so that backend refuses the model instead of simulating different physics.
4. Add `tests/test_<animal>.py` (an `"mjc"` episode, MJX matching `"mjc"`, a render smoke test), an `examples/`
   folder and a README.

## Migrating from 0.4.x

Version 1.0 is a rewrite and is not backwards compatible.

- BRT no longer depends on `moojoco`, `fprs` or `dm_control`. Models are built with `mujoco.MjSpec`, specifications
  are plain dataclasses instead of `fprs` specifications, and morphologies and arenas subclass
  `biorobot.core.Morphology` / `biorobot.core.Arena` instead of `moojoco`'s `MJCFMorphology` / `MJCFArena`.
- Module paths changed: e.g. `biorobot.brittle_star.mjcf.morphology.morphology` is now
  `biorobot.brittle_star.morphology`, and the per-backend `mjc_env.py` / `mjx_env.py` / `dual.py` modules are replaced
  by one class per task in `biorobot.<animal>.environment`. The simplest way to create an environment is now
  `biorobot.make("<animal>/<task>", backend=...)`.
- The environment API is functional on every backend: `state = env.reset(rng)` and `state = env.step(state, action)`.
  Use `biorobot.core.gym.GymnasiumEnv` for the Gymnasium API.
- Rewards of `"mjc"` environments were always 0 in 0.4.x; they are correct now, so results are not comparable.

To keep running code written for 0.4.x, pin the old version: `pip install biorobot==0.4.2`.

## Examples

Every animal has runnable examples, e.g.

```bash
python -m biorobot.brittle_star.examples.light_escape --backend mjx --render human
python -m biorobot.jumping_spider.examples.scripted_jump
python -m biorobot.seahorse.examples.open_loop
python -m biorobot.swimmer.examples.van_der_pol_cpg
```

The Jupyter notebooks of the [SEL3 course](https://github.com/Co-Evolve/SEL3-2024) use BRT 0.4.x
(`pip install biorobot==0.4.2`).

## Contributing

Interested in having a model developed? Please fill in this [form](https://forms.gle/h1xgnZA39h2qhX8k9). Bug reports
and pull requests are welcome on [GitHub](https://github.com/Co-Evolve/brt). Run `pytest -m "not slow"` and
`black biorobot tests scripts` before opening a pull request.

## Publications

Papers that use BRT's animals (the animal READMEs have the BibTeX):

- **Brittle star**:
  - *Decentralized neural dynamics and sensory constraints shape brittle star locomotion*, Marzougui, Adriaens and
    wyffels, Journal of the Royal Society Interface ([2026](https://doi.org/10.1098/rsif.2025.1347))
  - *Stability-Driven CPG Design for More Effective Quality-Diversity in Evolutionary Robotics*, Pex, Marzougui and
    wyffels, GECCO ([2026](https://doi.org/10.1145/3795095.3805096))
- **Seahorse tail**:
  - *Exploring the evolutionary adaptations of the unique seahorse tail's muscle architecture through in silico
    modelling and robotic prototyping*, Marzougui, Das, Mazzolai, Adriaens and wyffels, Journal of the Royal Society
    Interface ([2025](https://doi.org/10.1098/rsif.2024.0876)); try its
    [muscle playground](https://co-evolve.github.io/seahorse-muscle-architecture/)

Using BRT in a paper? Open a pull request to add it here.

## Citation

If you use BRT in your research, please cite this repository:

```bibtex
@software{marzougui_brt,
  author = {Marzougui, Dries},
  title = {{BRT}: The Bio-Inspired Robotics Testbed},
  url = {https://github.com/Co-Evolve/brt},
  version = {1.1.0},
  year = {2026}
}
```
