"""
sweep.py — the 100-simulation parameter sweep that fixed the parameter and
descriptor scales and serves as the MAP-Elites seed population.

Copied from MuJoCo-demo ``src/run_sweep.py`` (commit f7ad026): a 10 x 10 grid
over (omega, w_fwd) with the other seven parameters drawn per cell from a
per-cell RNG whose seed is drawn from the master RNG. Unchanged sampling
order. Settings now come from a config dict instead of a module constant.
"""

import sys
import time

import numpy as np
import pandas as pd

from .episode import evaluate


def make_param_table(cfg: dict) -> pd.DataFrame:
    """Build the omega_n*w_fwd_n parameter table. Deterministic given master_seed."""
    b = cfg["bounds"]
    g = cfg["grid"]
    master = np.random.default_rng(cfg["master_seed"])
    omega_values = np.linspace(b["omega"][0], b["omega"][1], g["omega_n"])
    w_fwd_values = np.linspace(b["w_fwd"][0], b["w_fwd"][1], g["w_fwd_n"])
    log_k_lo = np.log10(b["stiffness"][0])
    log_k_hi = np.log10(b["stiffness"][1])

    rows = []
    run_id = 0
    for w_fwd_idx, w_fwd in enumerate(w_fwd_values):
        for omega_idx, omega in enumerate(omega_values):
            seed = int(master.integers(0, 2**31 - 1))
            cell = np.random.default_rng(seed)
            rows.append(
                {
                    "run_id": run_id,
                    "omega_idx": omega_idx,
                    "w_fwd_idx": w_fwd_idx,
                    "omega": float(omega),
                    "w_fwd": float(w_fwd),
                    "N": int(cell.choice(b["N_choices"])),
                    "taper": float(cell.uniform(b["taper"][0], b["taper"][1])),
                    "stiffness": float(10 ** cell.uniform(log_k_lo, log_k_hi)),
                    "mu": float(cell.uniform(b["mu"][0], b["mu"][1])),
                    "w_bwd": float(cell.uniform(b["w_bwd"][0], b["w_bwd"][1])),
                    "amplitude_scale": float(
                        cell.uniform(b["amplitude_scale"][0], b["amplitude_scale"][1])
                    ),
                    "amp_taper": float(
                        cell.uniform(b["amp_taper"][0], b["amp_taper"][1])
                    ),
                    "seed": seed,
                }
            )
            run_id += 1
    return pd.DataFrame(rows)


def run_sweep(cfg: dict, n_override: int | None = None) -> tuple[pd.DataFrame, list]:
    """Run the sweep. Returns (results table, per-run wall times in s).

    `n_override` truncates the parameter table (used by the smoke test).
    """
    params = make_param_table(cfg)
    if n_override is not None:
        params = params.iloc[:n_override].copy()

    total = len(params)
    results, walltimes = [], []
    for _, row in params.iterrows():
        print(
            f"[{int(row['run_id']):>3}/{total}] "
            f"omega={row['omega']:.2f} w_fwd={row['w_fwd']:.2f}  "
            f"N={int(row['N'])} taper={row['taper']:.2f} "
            f"k={row['stiffness']:.4g} mu={row['mu']:.2f} "
            f"wb={row['w_bwd']:.2f} amp={row['amplitude_scale']:.2f} "
            f"at={row['amp_taper']:+.2f}",
            flush=True,
        )
        t0 = time.perf_counter()
        r = evaluate(row, cfg["episode"])
        walltimes.append(time.perf_counter() - t0)
        if r["failed"]:
            print(
                f"  run {int(row['run_id']):>3} FAILED ({r['failure']})",
                file=sys.stderr,
            )
        results.append(r)

    df = pd.concat([params.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    return df, walltimes
