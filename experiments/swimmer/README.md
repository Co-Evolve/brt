# swimmer — illuminating a swimmer's behaviour space

Code for the illustrative box *"Illuminating a swimmer's behaviour space"* in
the review on physics-based simulation of biological locomotion (*Physics of
Life Reviews*). It runs one parameter sweep and two MAP-Elites searches on an
abstract swimmer. The searches are identical except for their descriptor
axes. It then draws the two-panel box figure.

The code is copied from the experiment repository MuJoCo-demo
(Kermit-UGent/MuJoCo-demo, commit `f7ad026`). On 2026-11-03 three modelling
corrections were applied to it: the post-erratum CPG warm-up and settling
rule, head-first speed as the quality, and rejection of evaluations in which
MuJoCo reports an instability. Everything else is unchanged. With the
corrections switched off, the code reproduces the May 2026 runs bit-for-bit.
`VERIFICATION.md` gives both checks and the old and new numbers side by side.

MuJoCo-demo is a private repository, available on request. The commit hashes
and `results/` paths cited here and in `VERIFICATION.md` refer to it.

## The experiment

- **Body.** A chain of N ∈ {4, 6, 8, 10, 12} capsules (radius 15 mm, density
  600 kg/m³). The head segment is 0.1 m long, and segment lengths change
  linearly to `taper` × 0.1 m at the tail. Hinges carry passive torsional springs
  (`stiffness`, damping 0.005, range ±1.5 rad) and position servos (kp = 2).
  The body moves in the horizontal plane with gravity off. Because the head
  segment length is fixed, N also sets body length (0.3–1.5 m).
- **Fluid.** MuJoCo's phenomenological fluid model with
  `fluidshape="ellipsoid"`, viscosity 0.1 Pa·s and density 1000 kg/m³. It
  applies drag, added-mass and lift terms to each capsule without resolving a
  flow field. The integrator is Euler with dt = 2 ms.
- **Controller.** A chain of N Van der Pol oscillators,
  ẍᵢ = μ(1 − xᵢ²)ẋᵢ − ω²xᵢ + w_fwd(xᵢ₋₁ − xᵢ) + w_bwd(xᵢ₊₁ − xᵢ), with
  x₀ = x_{N+1} = 0. It is integrated with RK2 and drives joint i with target
  ampᵢ·xᵢ, where ampᵢ = amplitude_scale · (1 + amp_taper · (i − (N+1)/2)/(N/2)).
- **Search space (9 parameters).** Morphology: N, taper ∈ [0.5, 1.5],
  stiffness ∈ [0.001, 0.1] N·m/rad (log scale). Controller: μ ∈ [0.5, 3],
  ω ∈ [3, 15] rad/s, w_fwd ∈ [0.5, 5], w_bwd ∈ [0, 3],
  amplitude_scale ∈ [0.2, 0.8], amp_taper ∈ [−0.6, 0.6].
- **Episode.** The cycle period is T = 2π/ω. The CPG is first integrated
  alone for max(5 s, 40 T) so that it is on its limit cycle. MuJoCo then
  runs for a settling time of max(2 s, 2 T), which is excluded, followed by a
  5 s measurement window. Across ω ∈ [3, 15] rad/s, the warm-up is 16.8–84 s,
  the settling time is 2.0–4.2 s and the episode is 7.0–9.2 s. This is the
  post-erratum rule of MuJoCo-demo `src/v4/v4_model.py`, with the window held
  at 5 s so that energy always integrates over the same span.
- **Instability.** MuJoCo resets the simulation silently when it detects a
  bad position, velocity or acceleration. Any evaluation with such a warning
  (or a restarted clock) counts as failed (`failure = "unstable"`) and never
  enters the grid. Each search's `summary.md` reports the count.
- **Sweep** (`configs/sweep.yaml`). 100 simulations: a 10 × 10 grid over
  (ω, w_fwd) with the other seven parameters drawn at random per cell. It
  is the seed population of both searches; failed seeds are dropped. The
  parameter bounds are kept as run and come from the preliminary sweep of
  15 May 2026 (MuJoCo-demo `results/sweep_vdp_amptaper_2026-05-15_122723`).
  That sweep used the same bounds and master seed, hence the same 100
  parameter vectors, with 10 s episodes, no settling and unsigned speed.
- **Searches** (`configs/search_a.yaml`, `configs/search_b.yaml`). Vanilla
  MAP-Elites on a 20 × 20 grid, seeded with the 100 sweep simulations, run
  for 5,000 iterations with master seed 42. Each iteration draws a parent
  uniformly from the filled cells, applies Gaussian mutation (with N redrawn
  with probability 0.2), clips the child to the bounds, and keeps it if its
  cell is empty or if it is faster than the incumbent.

## Quantities

All quantities are computed over the 5 s measurement window (2500 steps,
starting after the settling time).

| quantity | definition |
|---|---|
| **speed** (m/s), signed, head-first | −(x_end − x_start) / 5 s. x is the position of the head segment `seg0` (the frame of the free joint) after the first and the last step of the window. The body extends from the head toward +x, so positive speed means the chain moves head first and negative speed means it moves tail first ("backward"). All capsules have the same radius; the head is defined by position, not width. |
| **quality** | speed (head-first, signed). A child replaces the incumbent only if it is strictly faster. |
| **energy** (J) | Σ_steps Σ_hinges abs(τ · ω̇) · dt. τ is the actuator torque (`qfrc_actuator`) and ω̇ the hinge angular velocity. Negative work counts as cost too. |
| **joint excursion** (rad) | Per hinge, the maximum minus the minimum angle over the window, then the mean over the N − 1 hinges. |
| lateral_amp, path_straightness, distance | Recorded but not used by the searches. `distance` is the raw +x displacement, so its sign is opposite to speed. |

**Binning.** Each descriptor maps linearly onto 20 bins between the bounds
in the config. Values outside the bounds go into the edge bin.

- **Search a**, x = log₁₀ speed over [log₁₀ 10⁻⁴, log₁₀ 0.5]. Before taking
  the logarithm, speed is replaced by max(speed, 10⁻⁴ m/s). Every solution
  slower than 10⁻⁴ m/s, backward (tail-first) swimmers included, therefore
  lands in the leftmost column, where forward swimmers can displace it. Speeds above
  0.5 m/s land in the rightmost column. y = log₁₀ energy over [log₁₀ 1, log₁₀ 500] J.
- **Search b**, x = joint excursion over [0.47, 2.06] rad (linear) and
  y = log₁₀ energy over [log₁₀ 10.93, log₁₀ 414.22] J. These are the rounded
  5th–95th percentiles of the preliminary sweep of 15 May 2026, not of
  `configs/sweep.yaml`.
- The search (a) bounds were set by hand on 15 May. They span the
  preliminary sweep's speed range (2.7×10⁻⁴ to 0.59 m/s) but are not
  percentiles of it, and their origin is not recorded.
- All descriptor bounds are kept as run.

## Install

Python 3.11 or later. Exact versions are in `requirements.txt`; the
experiment ran on 3.12. Only the C MuJoCo bindings are used (no MJX, no JAX).

This folder needs its own environment. It pins `mujoco==3.8.1`, which the
results depend on, while the `biorobot` package at the root of this
repository requires `mujoco>=3.17`. Do not install it into the `biorobot`
environment. Nothing here imports `biorobot`. All commands below are run
from this folder.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m playwright install chromium   # only for PNG/PDF rendering
```

## Reproduce everything

```bash
.venv/bin/python scripts/reproduce_all.py --jobs 2
```

This runs the sweep, then both searches (in parallel with `--jobs 2`), then
a read-only analysis of search b (`scripts/excursion_by_energy.py`, about
1 min), and finally writes `output/figure/box_descriptor_choice.svg`. Expected runtime on an
Apple M3 CPU: about 30 s for the sweep, then about 30 min for the two
searches with `--jobs 2` (search a takes 30 min, search b 21 min) or about
51 min with `--jobs 1`. Most of the time goes into the long CPG warm-ups.
Everything runs single-threaded on the CPU; no GPU is used.

Outputs, all under `output/`:

```
sweep/       data.parquet, config.yaml, timing.json
search_a/    genealogy.parquet (every evaluated solution), grid_final.parquet,
search_b/      config.yaml, summary.md, progress_log.txt, timing.json
analysis/    excursion_by_energy.md (per energy row of search b: excursion of
               the fastest elite, Spearman rho of excursion vs speed)
figure/      box_descriptor_choice.svg (+ .pdf, .png, cvd_sheet.png once rendered)
checksums.txt  SHA-256 of the five result tables, for run-to-run comparison
```

Only `output/figure/` is committed: the SVG, PDF, PNG and CVD sheet of the
figure for the manuscript. Everything else under `output/` is regenerated by
`reproduce_all.py`.

Result tables contain no wall-clock data, so two runs on the same platform
give identical files. Results on another CPU architecture or MuJoCo build
may differ in the last bits.

Render the figure and check it:

```bash
.venv/bin/python scripts/render.py output/figure/box_descriptor_choice.svg output/figure/box_descriptor_choice.png 2
.venv/bin/python scripts/render_pdf.py output/figure/box_descriptor_choice.svg output/figure/box_descriptor_choice.pdf
.venv/bin/python scripts/cvd.py output/figure/box_descriptor_choice.png output/figure/cvd_sheet.png
```

Smoke test (about 5 s): `.venv/bin/python -m pytest`.

## Layout

```
swimmer/      model.py (MuJoCo XML), cpg.py (Van der Pol chain),
              episode.py (rollout, descriptors, evaluate), sweep.py,
              mapelites.py, runs.py (config loading, output folders)
configs/      sweep.yaml, search_a.yaml, search_b.yaml: every bound and seed;
              excursion_by_energy.yaml: settings and seed of the analysis
scripts/      reproduce_all.py, make_figure.py, excursion_by_energy.py;
              figure tooling: house.py (house style), render.py,
              render_pdf.py, cvd.py
tests/        test_smoke.py
```

`house.py`, `render.py` and `cvd.py` are copies of the shared figure tooling,
so `black` skips them (see `pyproject.toml`). `house.py` has been verified
byte-identical to its source. `render.py` and `cvd.py` were transcribed from
the authors' copy and have not been diffed against the originals.

## Scope and caveats

- The fluid model is phenomenological. No hydrodynamic claim is made, and
  all comparisons are internal to this model.
- The two searches share their 100 seeds and all settings, and diverge from
  the first iteration because parents are drawn from filled cells. They are
  two searches, not one population binned two ways.
- The manuscript's May 2026 numbers came from the uncorrected version: a
  fixed 5 s warm-up and 1 s settling, +x speed and no instability check.
  VERIFICATION.md lists both sets of numbers.

MIT licence (as the brt repository).
