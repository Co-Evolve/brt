"""
mapelites.py — vanilla Mouret-Clune MAP-Elites on the swimmer.

Copied from MuJoCo-demo ``src/mapelites.py`` (search a) and
``src/mapelites_gait_cost.py`` (search b), commit f7ad026. The two original
drivers were line-for-line identical apart from the binning functions and the
output folder name, so they are merged here into one driver that takes the
descriptor specification from the config. Mutation, parent selection,
replacement rule, seeding and the order of every RNG call are unchanged.

Descriptor kinds:
  "log_speed_x_log_energy"   (search a)
      x = log10(max(speed, speed_clip_min)) over [speed_min_log, speed_max_log]
      y = log10(energy)                      over [energy_min_log, energy_max_log]
  "excursion_x_log_energy"   (search b)
      x = joint_excursion (rad, linear)      over [d1_min, d1_max]
      y = log10(energy)                      over [d2_min_log, d2_max_log]
Values outside the bounds are clamped into the edge cells. Quality is signed
head-first speed in both searches (see episode.py).
"""

import time

import numpy as np
import pandas as pd

from .episode import DESCRIPTOR_NAMES, PARAM_NAMES, evaluate

# --- binning --------------------------------------------------------------


def _edge_clamp(frac: float, n: int) -> int:
    idx = int(np.floor(frac * n))
    return max(0, min(n - 1, idx))


def bin_speed_index(speed: float, d: dict, n: int) -> int:
    s = max(speed, d["speed_clip_min"])
    log_s = np.log10(s)
    frac = (log_s - d["speed_min_log"]) / (d["speed_max_log"] - d["speed_min_log"])
    return _edge_clamp(frac, n)


def bin_log_energy_index(energy: float, lo_log: float, hi_log: float, n: int) -> int:
    if energy <= 0:
        return 0
    log_e = np.log10(energy)
    frac = (log_e - lo_log) / (hi_log - lo_log)
    return _edge_clamp(frac, n)


def bin_excursion_index(je: float, d: dict, n: int) -> int:
    """Linear binning over joint_excursion; out-of-bounds -> nearest edge."""
    frac = (je - d["d1_min"]) / (d["d1_max"] - d["d1_min"])
    return _edge_clamp(frac, n)


def cell_of(descriptors: dict, d: dict, n: int) -> tuple[int, int]:
    """(cx, cy) grid cell for one solution under descriptor spec `d`."""
    if d["kind"] == "log_speed_x_log_energy":
        cx = bin_speed_index(descriptors["speed"], d, n)
        cy = bin_log_energy_index(
            descriptors["energy"], d["energy_min_log"], d["energy_max_log"], n
        )
    elif d["kind"] == "excursion_x_log_energy":
        cx = bin_excursion_index(descriptors["joint_excursion"], d, n)
        cy = bin_log_energy_index(
            descriptors["energy"], d["d2_min_log"], d["d2_max_log"], n
        )
    else:
        raise ValueError(f"unknown descriptor kind {d['kind']!r}")
    return cx, cy


# --- mutation -------------------------------------------------------------


def _mut_clip(parent_val, sigma, lo, hi, rng):
    return float(np.clip(parent_val + rng.normal(0.0, sigma), lo, hi))


def mutate(parent: dict, rng: np.random.Generator, cfg: dict) -> dict:
    b, s = cfg["bounds"], cfg["mutation"]
    p = dict(parent)
    if rng.random() < s["p_N_resample"]:
        p["N"] = int(rng.choice(b["N_choices"]))
    p["taper"] = _mut_clip(
        parent["taper"], s["sigma_taper"], b["taper"][0], b["taper"][1], rng
    )
    log_k = float(np.log10(parent["stiffness"]))
    log_k_new = float(
        np.clip(
            log_k + rng.normal(0.0, s["sigma_log_stiffness"]),
            np.log10(b["stiffness"][0]),
            np.log10(b["stiffness"][1]),
        )
    )
    p["stiffness"] = float(10.0**log_k_new)
    p["mu"] = _mut_clip(parent["mu"], s["sigma_mu"], b["mu"][0], b["mu"][1], rng)
    p["omega"] = _mut_clip(
        parent["omega"], s["sigma_omega"], b["omega"][0], b["omega"][1], rng
    )
    p["w_fwd"] = _mut_clip(
        parent["w_fwd"], s["sigma_w_fwd"], b["w_fwd"][0], b["w_fwd"][1], rng
    )
    p["w_bwd"] = _mut_clip(
        parent["w_bwd"], s["sigma_w_bwd"], b["w_bwd"][0], b["w_bwd"][1], rng
    )
    p["amplitude_scale"] = _mut_clip(
        parent["amplitude_scale"],
        s["sigma_amplitude_scale"],
        b["amplitude_scale"][0],
        b["amplitude_scale"][1],
        rng,
    )
    p["amp_taper"] = _mut_clip(
        parent["amp_taper"],
        s["sigma_amp_taper"],
        b["amp_taper"][0],
        b["amp_taper"][1],
        rng,
    )
    return p


# --- grid insertion -------------------------------------------------------


def try_insert(grid, qual, solution_id, descriptors, cfg):
    """Replacement rule: new beats incumbent iff cell empty OR new_q > old_q.

    Returns (cx, cy, replaced_incumbent, became_incumbent).
    """
    cx, cy = cell_of(descriptors, cfg["descriptors"], cfg["grid_n"])
    new_q = descriptors["speed"]
    if grid[cy, cx] == -1:
        grid[cy, cx] = solution_id
        qual[cy, cx] = new_q
        return cx, cy, False, True
    if new_q > qual[cy, cx]:
        grid[cy, cx] = solution_id
        qual[cy, cx] = new_q
        return cx, cy, True, True
    return cx, cy, False, False


# --- driver ---------------------------------------------------------------


def run_mapelites(cfg: dict, seeds_df: pd.DataFrame) -> dict:
    """Run one search. Returns genealogy, final grid, timings and a log.

    `seeds_df` is the sweep table; failed rows (including unstable ones) are
    dropped, as in the original, so they never enter the grid. Wall times are returned separately from the genealogy so the
    result tables are bit-for-bit reproducible.
    """
    seeds_df = seeds_df[~seeds_df.failed].reset_index(drop=True)
    n_seeds = len(seeds_df)

    grid_n = cfg["grid_n"]
    grid = np.full((grid_n, grid_n), -1, dtype=int)
    qual = np.full((grid_n, grid_n), -np.inf, dtype=float)
    iter_first_fill = np.full((grid_n, grid_n), -1, dtype=int)

    genealogy: list[dict] = []
    rng = np.random.default_rng(cfg["master_seed"])

    # Phase 1: seed the grid from the sweep
    for _, row in seeds_df.iterrows():
        sid = len(genealogy)
        params = {p: row[p] for p in PARAM_NAMES}
        params["N"] = int(params["N"])
        descriptors = {d: float(row[d]) for d in DESCRIPTOR_NAMES}
        cx, cy, replaced, became = try_insert(grid, qual, sid, descriptors, cfg)
        if became and iter_first_fill[cy, cx] == -1:
            iter_first_fill[cy, cx] = 0
        genealogy.append(
            {
                "solution_id": sid,
                "parent_id": None,
                "generation": 0,
                "iteration": None,
                "cell_x_idx": int(cx),
                "cell_y_idx": int(cy),
                "replaced_incumbent": bool(replaced),
                "became_incumbent": bool(became),
                **params,
                **descriptors,
                "quality": float(descriptors["speed"]),
                "failure": None,
            }
        )

    n_filled_after_seeds = int((grid != -1).sum())
    log = [
        f"seeded {n_seeds} sweep rows; "
        f"{n_filled_after_seeds}/{grid_n * grid_n} cells filled"
    ]
    print(log[-1], flush=True)

    # Phase 2: MAP-Elites iterations
    walltimes = []
    for it in range(1, cfg["n_iterations"] + 1):
        t0 = time.perf_counter()

        filled_idx = np.argwhere(grid != -1)
        cy_p, cx_p = filled_idx[rng.integers(0, len(filled_idx))]
        parent_sid = int(grid[cy_p, cx_p])
        parent = {p: genealogy[parent_sid][p] for p in PARAM_NAMES}
        child = mutate(parent, rng, cfg)
        descriptors = evaluate(child, cfg["episode"])
        sid = len(genealogy)
        generation = int(genealogy[parent_sid]["generation"]) + 1

        if descriptors["failed"]:
            genealogy.append(
                {
                    "solution_id": sid,
                    "parent_id": parent_sid,
                    "generation": generation,
                    "iteration": it,
                    "cell_x_idx": -1,
                    "cell_y_idx": -1,
                    "replaced_incumbent": False,
                    "became_incumbent": False,
                    **child,
                    **{d: np.nan for d in DESCRIPTOR_NAMES},
                    "quality": np.nan,
                    "failure": descriptors["failure"],
                }
            )
        else:
            cx, cy, replaced, became = try_insert(grid, qual, sid, descriptors, cfg)
            if became and iter_first_fill[cy, cx] == -1:
                iter_first_fill[cy, cx] = it
            genealogy.append(
                {
                    "solution_id": sid,
                    "parent_id": parent_sid,
                    "generation": generation,
                    "iteration": it,
                    "cell_x_idx": int(cx),
                    "cell_y_idx": int(cy),
                    "replaced_incumbent": bool(replaced),
                    "became_incumbent": bool(became),
                    **child,
                    **{d: float(descriptors[d]) for d in DESCRIPTOR_NAMES},
                    "quality": float(descriptors["speed"]),
                    "failure": None,
                }
            )

        walltimes.append(time.perf_counter() - t0)
        if it % cfg["print_every"] == 0:
            valid_q = qual[grid != -1]
            line = (
                f"iter {it:>4}: {int((grid != -1).sum()):>3}/{grid_n * grid_n} "
                f"cells, max_q={valid_q.max():+.4f}, mean_q={valid_q.mean():+.4f}"
            )
            print(line, flush=True)
            log.append(line)

    grid_rows = []
    for y in range(grid_n):
        for x in range(grid_n):
            if grid[y, x] != -1:
                grid_rows.append(
                    {
                        "cell_x_idx": int(x),
                        "cell_y_idx": int(y),
                        "solution_id": int(grid[y, x]),
                        "quality": float(qual[y, x]),
                        "first_fill_iter": int(iter_first_fill[y, x]),
                    }
                )
    return {
        "genealogy": pd.DataFrame(genealogy),
        "grid": pd.DataFrame(grid_rows),
        "walltimes": np.asarray(walltimes),
        "n_seeds": n_seeds,
        "n_filled_after_seeds": n_filled_after_seeds,
        "log": log,
    }
