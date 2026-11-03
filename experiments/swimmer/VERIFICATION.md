# Verification

Date: 2026-11-03. Machine: Apple M3 (macOS 25.6, arm64), Python 3.12.2,
MuJoCo 3.8.1, NumPy 2.4.4 (full pins in `requirements.txt`).

## 1. Provenance

- **Source:** MuJoCo-demo (Kermit-UGent/MuJoCo-demo), commit
  `f7ad026ba381750eebbc48df86d6616884aacbe3` (HEAD on 2026-11-03, clean
  working tree).
- **Identity with the May runs.** In that commit `src/vdp_network.py`,
  `src/mapelites.py`, `src/mapelites_gait_cost.py` and `src/run_sweep.py`
  are byte-identical to commit `55c2bd2` (2026-05-15), the commit that
  holds the code of the runs reported in the manuscript.
  `src/swimmer.py` differs from `55c2bd2` only by a radius parameter
  (default unchanged), helper functions and the C. elegans preset.
- **Runs of record** (MuJoCo-demo `results/`): sweep
  `sweep_vdp_amptaper_2026-05-15_125319`; search a
  `mapelites_2026-05-15_130056`; search b
  `mapelites_gait_cost_2026-05-15_133456`.

## 2. Reproduction of the manuscript runs (code as in May)

The package was first built with no modelling change: fixed 5 s CPG
warm-up, 1 s settling, 6 s episode, +x speed, no instability check. Three
reproductions were run from clean copies: one in a development environment
and two (#2, #3) in a fresh environment installed only from the pins.

- **Run to run.** The SHA-256 of all five result tables (sweep,
  2 × genealogy, 2 × final grid) is identical across all three runs.
- **Against the archived May parquets.** Every result column is
  bit-identical: sweep 100 × 20; each genealogy 5,100 × 24; final grids
  307 × 5 and 381 × 5. Wall time is the only column compared away, because
  the package stores it separately.

| quantity | manuscript | obtained (May code) | match |
|---|---|---|---|
| maximum forward speed, search a | +0.71 m/s | +0.7138 m/s | yes |
| maximum forward speed, search b | +0.74 m/s | +0.7385 m/s | yes |
| filled cells, search a | 307 of 400 | 307 of 400 | yes |
| filled cells, search b | 381 of 400 | 381 of 400 | yes |
| high-energy, low-excursion corner of b | 4 of 16 | 4 of 16 | yes |
| backward-swimming elites in b | two, −0.42 and −0.36 m/s | two, −0.4244 (cell 3,18) and −0.3563 m/s (cell 4,19) | yes |
| parameter vectors evaluated by both searches | 2 of 5,000 (0.04%) | 2 of 5,000 (0.04%); first divergent genealogy row 100 = iteration 1 | yes |

**Warm-up: the manuscript runs predate the correction.** The
manuscript's runs used a fixed 5 s CPG warm-up and 1 s settling. The
post-erratum rule (MuJoCo-demo `src/v4/v4_model.py`, 2026-08-25) is
warm-up = max(5 s, 40 cycles) and settling = max(2 s, 2 cycles). Over this
experiment's ω ∈ [3, 15] rad/s, that lengthens the warm-up for every
evaluation (16.8–84 s).

As a diagnostic, the 688 elites of record were re-evaluated under the
corrected warm-up only:

- Only 33 of 307 (a) and 89 of 381 (b) stayed in their cell.
- 73 elites of (a) changed the sign of their speed.
- The median relative speed change was 54% (a) and 12% (b).
- The maximum speeds barely moved.

**Further findings on the May code**

- **Direction.** The head segment `seg0` sits at x = 0 and the chain extends
  toward +x, so May's "forward" (+x) speed was tail-first. The top May
  elites had tail-to-head body waves and moved tail first. The two
  "backward" elites (−0.42 and −0.36 m/s) had head-to-tail waves and moved
  head first.
- **Silent resets.** MuJoCo resets the state silently on a bad acceleration.
  Re-running all 10,200 May evaluations with a warning check found
  3 of 5,100 resets in search a (all N = 4 and near-stationary; one of them,
  in cell (3, 12), was an elite) and 0 of 5,100 in search b. The
  re-evaluation reproduced every recorded speed exactly.
- **Descriptor bounds.** The search b bounds (0.47–2.06 rad,
  10.93–414.22 J) are the rounded 5th–95th percentiles of the preliminary
  sweep of 15 May (`sweep_vdp_amptaper_2026-05-15_122723`: 10 s episodes,
  no settling, unsigned speed). They are not percentiles of the seed sweep,
  which gives 0.43–1.81 rad and 4.5–209 J. The search a bounds were set by
  hand.

These findings led to the decisions in §3.

## 3. Corrected configuration (decisions of 2026-11-03)

1. **Warm-up and settling:** the post-erratum rule above, with a 5 s
   measurement window, so the episode lasts settling + 5 s (7.0–9.2 s).
2. **Quality:** head-first speed, −Δx/5 s, positive towards `seg0`. Only
   the sign changed. Search a bins the same speed variable.
3. **Instability:** any evaluation with a MuJoCo bad-qpos, bad-qvel or
   bad-qacc warning, or with a restarted clock, is invalid and never enters
   the grid.
4. **Bounds:** parameter and descriptor bounds unchanged; their provenance
   is documented in the configs and the README.
5. **Dependencies:** JAX and MJX removed. Nothing imported them.

`PREDICTIONS.md` was written before these runs. Its SHA-256 then was
`920928858960e9048b9a09cfeca2ef4b6f324ea5927e48b1a40f3807a7d0b7d9`, and it
is unchanged.

**Two runs from clean checkouts**, fresh environment from
`requirements.txt`, `reproduce_all.py --jobs 2`, run in parallel. Each took
30.4 min wall time. The result tables are byte-identical:

```
536f900e4b420a81a12b8eff8c3ee5d067f07ef522be1966196767381eb580a9  sweep/data.parquet
90170b0d7e05b242b658f522a169ab3b1de5497adc86e212653183c8e3a08ae6  search_a/genealogy.parquet
fc5a16aed91e52db63051865dd0bcbeab97d9e85543df9204a6d5a1b501b3c94  search_a/grid_final.parquet
ecdc85b697dc318328f0dc40255f2e3afc1b4d0bd37d4caed1c60b3f6d7b9ff5  search_b/genealogy.parquet
d9b2aa15911bec32befd60bbf7a2e2c84958332f70db8fdff37fd55db170a968  search_b/grid_final.parquet
```

The SVGs written by the two runs were also identical. `output/` holds
run #1's tables, with the figure regenerated by the final `make_figure.py`
(changed after the runs; no simulation involved).

### Every quantity in the caption and box: old and new

"Old" is the manuscript value, reproduced exactly in §2. "New" is the
corrected configuration. Speeds are signed head-first speed in the new
column and +x speed in the old one, so old "backward" and new "backward"
are different physical motions.

| quantity | old (manuscript) | new (corrected) | changed? |
|---|---|---|---|
| seed population | 100 simulations | 100 simulations, 0 failed, 0 unstable | no |
| budget, grid, master seed | 5,000 iter., 20 × 20, seed 42 | same | no |
| maximum speed, search a | +0.71 m/s | **+0.66 m/s** (+0.6611) | yes |
| maximum speed, search b | +0.74 m/s | **+0.66 m/s** (+0.6626) | yes |
| maxima near-identical | yes | yes (differ by 0.0015 m/s) | no |
| filled cells, search a | 307 of 400 | **292 of 400** | yes |
| filled cells, search b | 381 of 400 | **382 of 400** | yes |
| clip column of a: elites with negative speed | 8, none below −0.014 m/s | **7, none below −0.082 m/s** | yes |
| clip column of a: forward swimmers displace nearly all backward ones | 8 of 1,244 negative evaluations survive | 7 of 609 survive | wording holds |
| high-energy, low-excursion corner of b | 4 of 16 | **5 of 16** (cells (2,16), (3,16), (3,17), (3,18), (3,19)) | yes; "largely empty" holds |
| backward-swimming elites in b | two, −0.42 and −0.36 m/s | **one, −0.0011 m/s** (cell 18,1; N = 4, 1.90 rad, 15.6 J): effectively stationary | **yes, claim does not survive** |
| negative-speed evaluations, a / b | 1,244 / 824 of 5,100 | 609 / 392 of 5,100 | yes |
| divergence | from iteration 1 | from iteration 1 (first divergent row 100) | no |
| parameter vectors evaluated by both | 2 of 5,000 (0.04%) | **1 of 5,000 (0.02%)** | yes |
| evaluations rejected as unstable, a / b | not checked (3 / 0 silent resets found afterwards) | **48 / 1** of 5,000 offspring; sweep 0 | new |
| cells filled by the seeds, a / b | 46 / 74 | 66 / 76 | yes |
| colour scale | forward speed, 0 → 0.74, viridis | head-first speed, −0.10 → +0.70, crimson | yes |

### Predictions (PREDICTIONS.md) against outcome

| prediction | outcome |
|---|---|
| Both searches still diverge from the first iteration and share almost no evaluated parameter vectors. | **Confirmed.** First divergent row 100 (iteration 1); 1 of 5,000 vectors shared. |
| Search b fills more cells than search a. | **Confirmed.** 382 vs 292. |
| Whether swimmers moving against the quality direction still hold cells of their own in search b: unknown. | **Only marginally.** One elite has negative speed, −0.0011 m/s: a body that barely moves, not a tail-first swimmer. No elite of b moves tail first at more than 1.1 mm/s. |
| Whether the high-energy, low-excursion corner of search b stays mostly empty: unknown. | **Yes.** 5 of 16 cells filled, 11 empty. |

## 4. Figure checks

- **SVG:** 1800 × 1005 units = 170 × 94.9 mm. Panels are the named groups
  `panel-a-speed-energy` and `panel-b-excursion-energy`; the legend is
  `legend-speed-scale`.
- **House style:** live Helvetica text in #4c4f69, #5c5f77, #7c7f93 and
  #bcc0cc. There are no `<marker>` elements (arrowheads are filled paths)
  and no black fills.
- **Colour:** one shared sequential scale, a single crimson hue (house
  behaviour colour), with OKLab lightness linear from 0.94 to 0.32. Neither
  blue #1e66f5 nor green #40a02b is used. White is used only for cells that
  were not reached.
- **Tail-first elites** carry a thick dashed ink outline, so they don't
  depend on colour alone.
- **CVD (`cvd.py`, deuteranopia and protanopia):** the ramp's lightness is
  monotone in all three views.
  - The ΔE between white and a speed-0 cell is 30.1 in normal vision, 20.1
    under deuteranopia and 20.6 under protanopia.
  - Between white and the lightest step (−0.10 m/s) it is 12.3 / 8.5 / 8.8.
    Those cells are outlined anyway.
  - Simulation sheet: `output/figure/cvd_sheet.png`.
- **Full-resolution PNG (2×):** inspected for overlapping text. The panel a
  clip note and the panel b speed label were checked in close-ups; nothing
  overlaps.
- **PDF:** vector, rendered by headless Chromium (`render_pdf.py`), with
  Helvetica embedded as text.

### Configuration-spaces figure (`make_spaces_figure.py`)

Checked against the authors' figure-tooling copy (`biosim_figures_house_style/scripts/`);
it was not run because its `glyphs` module is missing. Its behaviour panel
borrows the box axes:

| element | spaces figure | new box figure | consistent? |
|---|---|---|---|
| x axis | "joint excursion" | "joint excursion (rad)" | yes (the schematic has no units) |
| y axis | "energy use (log scale)" | "energy (J, log scale)" | **no**: "energy use" vs "energy" |
| colour | "best forward speed in the cell" | "head-first speed (m/s)" | **no**: "forward" vs "head-first" |
| colour hue | crimson tints | crimson ramp | yes |
| corner annotation | "never reached" (docstring: "the high-energy, low-excursion corner is never reached") | 5 of 16 cells reached; "not reached by this search" | **no**: overclaims. The data have 5 of 16 corner cells reached, and the box avoids "never/unreachable" |
| grid | 19 × 19 | 20 × 20 | schematic; minor |

## 5. Differences from the original code

| difference | reason |
|---|---|
| Warm-up max(5 s, 40 cycles), settling max(2 s, 2 cycles), episode = settling + 5 s | Decision 1: post-erratum correction; the 5 s window is kept so that energy always integrates over the same span |
| Speed = −Δx / window (head-first); `distance` keeps the +x sign | Decision 2 |
| MuJoCo instability warnings or a restarted clock make an evaluation invalid (`failure = "unstable"`); counts in `summary.md` | Decision 3 |
| `failure` column in the sweep table and genealogies | Records why an evaluation failed |
| Model compiled with `MjModel.from_xml_string`, no XML cache under `data/` and no per-simulation JSON records | No writes outside `output/`; verified bit-identical results (§2) |
| The two search scripts merged into one driver; the descriptor kind comes from the config | They were line-for-line identical apart from binning; RNG call order unchanged (verified, §2) |
| Settings read from YAML configs instead of module-level `CONFIG` dicts; the seed sweep is named explicitly instead of "latest `sweep_*` folder" | All randomness and bounds in the configs; no glob over timestamped folders |
| Fixed output folders (`output/<run>/`) instead of timestamped `results/` folders | Reproducible paths, no absolute paths |
| Wall times moved to `timing.json`; sweep CSV copy dropped | Result tables are byte-reproducible |
| Rendering branch (mediapy frames) removed from `simulate` | No video output in scope |
| C. elegans preset, worm code, v3/v4 code, morphology runs, notebooks, render scripts not copied | Out of scope: only the sweep and the two searches of the box are packaged |
| New figure script (`make_figure.py`, house style) replaces `src/box_figure.py` (matplotlib) | The box figure follows the review's house style (live-text SVG) |
| JAX and MJX removed from the pins | Decision 5: not imported |
| Formatted with black 26.10.1; `house.py`, `render.py` and `cvd.py` excluded | Repository formatting; the shared figure tooling is kept as in its source (`house.py` verified byte-identical; `render.py`, `cvd.py` not diffed) |

## 6. Claim in box sentence 4

Question: does "at a given energy, large joint excursions buy no extra speed"
hold across the energy rows of search b (corrected configuration)?

`scripts/excursion_by_energy.py` answers it. Settings are in
`configs/excursion_by_energy.yaml`; `reproduce_all.py` runs it after the
searches. It reads `output/search_b/` only and writes
`output/analysis/excursion_by_energy.md`. For each energy row with at least
5 filled cells, it reports the excursion of the fastest elite and the
Spearman correlation between excursion and speed across that row's elites.

| row | energy (J) | n | excursion range (rad) | fastest elite: excursion (rad) | its rank by excursion | its speed (m/s) | speed at largest excursion (m/s) | Spearman ρ | p |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 10.9–13.1 | 16 | 0.48–1.79 | 0.48 | 1 | 0.204 | 0.002 | −0.97 | <0.001 |
| 1 | 13.1–15.7 | 20 | 0.49–2.16 | 0.49 | 1 | 0.240 | 0.008 | −0.94 | <0.001 |
| 2 | 15.7–18.9 | 19 | 0.51–2.40 | 0.51 | 1 | 0.277 | 0.016 | −0.97 | <0.001 |
| 3 | 18.9–22.6 | 20 | 0.49–1.98 | 0.49 | 1 | 0.306 | 0.005 | −0.99 | <0.001 |
| 4 | 22.6–27.1 | 20 | 0.46–2.15 | 0.46 | 1 | 0.336 | 0.028 | −0.94 | <0.001 |
| 5 | 27.1–32.5 | 20 | 0.49–1.99 | 0.49 | 1 | 0.348 | 0.026 | −0.99 | <0.001 |
| 6 | 32.5–39.0 | 20 | 0.51–2.10 | 0.57 | 2 | 0.371 | 0.062 | −0.99 | <0.001 |
| 7 | 39.0–46.8 | 20 | 0.50–2.08 | 0.59 | 2 | 0.389 | 0.098 | −0.98 | <0.001 |
| 8 | 46.8–56.1 | 20 | 0.54–2.20 | 0.56 | 2 | 0.420 | 0.122 | −0.97 | <0.001 |
| 9 | 56.1–67.3 | 20 | 0.54–2.04 | 0.79 | 5 | 0.444 | 0.134 | −0.98 | <0.001 |
| 10 | 67.3–80.7 | 20 | 0.55–2.06 | 0.70 | 3 | 0.455 | 0.161 | −0.97 | <0.001 |
| 11 | 80.7–96.8 | 20 | 0.49–1.99 | 0.79 | 5 | 0.484 | 0.175 | −0.91 | <0.001 |
| 12 | 96.8–116.1 | 20 | 0.55–2.04 | 0.93 | 6 | 0.508 | 0.225 | −0.94 | <0.001 |
| 13 | 116.1–139.2 | 20 | 0.55–2.06 | 0.79 | 5 | 0.541 | 0.262 | −0.87 | <0.001 |
| 14 | 139.2–166.9 | 20 | 0.55–2.01 | 0.97 | 7 | 0.574 | 0.255 | −0.81 | <0.001 |
| 15 | 166.9–200.2 | 19 | 0.62–1.99 | 0.92 | 5 | 0.585 | 0.290 | −0.83 | <0.001 |
| 16 | 200.2–240.1 | 18 | 0.71–1.99 | 0.97 | 5 | 0.602 | 0.301 | −0.80 | <0.001 |
| 17 | 240.1–288.0 | 17 | 0.76–1.99 | 1.27 | 8 | 0.633 | 0.330 | −0.84 | <0.001 |
| 18 | 288.0–345.4 | 17 | 0.76–2.01 | 1.28 | 8 | 0.650 | 0.290 | −0.68 | 0.003 |
| 19 | 345.4–414.2 | 16 | 0.78–1.90 | 0.94 | 3 | 0.663 | 0.427 | −0.65 | 0.008 |

Rows analysed: 20 (at least 5 filled cells). ρ < 0 in 20, ρ > 0 in 0; ρ < 0 with p < 0.05 in 20. Fastest elite in the row's lowest-excursion cell: 6 rows. p: two-sided permutation test, 20,000 shuffles, seed 0.

**Result.** The claim holds in every row, and it understates the pattern.
Within each energy row, larger excursion goes with *lower* speed:

- ρ ranges from −0.65 to −0.99 across all 20 rows. No row has a positive
  correlation.
- **Below about 27 J (rows 0–5):** the fastest elite sits in the row's
  lowest-excursion cell.
- **At higher energies:** the fastest elite sits at an intermediate excursion
  (0.7–1.3 rad). Up to that point more excursion does buy speed; beyond it,
  speed falls.
- So "large excursions buy no extra speed" is true in every row, and "large
  excursions cost speed" is also supported. "Excursion never helps" would be
  false above about 50 J.

**Caveats**

1. **These are elites, not the population.** Each cell holds the fastest
   solution found there, so the numbers describe the illuminated frontier,
   not typical behaviour.
2. **The p-values are descriptive.** The elites of a row are not independent
   samples (neighbouring elites often share lineages), so read p as a measure
   of consistency, not as an inferential test.
3. **Edge cells hold clamped values.** Excursion is the elite's measured
   value, and the edge cells hold values beyond the axis bounds (for example
   2.40 and 0.46 rad). That is why some row ranges exceed 0.47–2.06 rad.
4. **Corrected run only.** This uses the corrected configuration of §3; the
   analysis was not repeated for the May runs.
