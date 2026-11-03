"""
runs.py — config loading and run-folder writing for the reproduction.

Each run writes to ``<out>/<run name>/``. Result tables contain no wall-clock
data, so they are bit-for-bit reproducible; wall times go to timing.json.
"""

import hashlib
import json
import platform
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import yaml

from .mapelites import run_mapelites
from .sweep import run_sweep


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def do_sweep(cfg: dict, out: Path) -> Path:
    run_dir = Path(out) / cfg["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    df, walltimes = run_sweep(cfg)
    df.to_parquet(run_dir / "data.parquet", index=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "timing.json").write_text(
        json.dumps(
            {"total_s": float(np.sum(walltimes)), "environment": _environment()},
            indent=2,
        )
    )
    n_unst = int((df["failure"] == "unstable").sum())
    print(
        f"sweep: {len(df)} runs, {int(df['failed'].sum())} failed "
        f"({n_unst} unstable) -> {run_dir}"
    )
    return run_dir


def do_search(cfg: dict, out: Path) -> Path:
    run_dir = Path(out) / cfg["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = pd.read_parquet(Path(out) / cfg["seed_population"] / "data.parquet")
    res = run_mapelites(cfg, seeds)
    res["genealogy"].to_parquet(run_dir / "genealogy.parquet", index=False)
    res["grid"].to_parquet(run_dir / "grid_final.parquet", index=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "progress_log.txt").write_text("\n".join(res["log"]) + "\n")
    (run_dir / "summary.md").write_text(_summary(cfg, res))
    w = res["walltimes"]
    (run_dir / "timing.json").write_text(
        json.dumps(
            {
                "total_s": float(w.sum()),
                "mean_per_iteration_s": float(w.mean()),
                "environment": _environment(),
            },
            indent=2,
        )
    )
    print(f"{cfg['run']}: {len(res['grid'])}/{cfg['grid_n'] ** 2} cells -> {run_dir}")
    return run_dir


def _summary(cfg: dict, res: dict) -> str:
    grid, gen = res["grid"], res["genealogy"]
    n_cells = cfg["grid_n"] ** 2
    offspring = gen[gen["iteration"].notna()]
    n_failed = int(offspring["quality"].isna().sum())
    n_unstable = int((offspring["failure"] == "unstable").sum())
    return (
        f"# {cfg['run']} — summary\n\n"
        f"Descriptors: `{cfg['descriptors']['kind']}`. Seeded from "
        f"`{cfg['seed_population']}` ({res['n_seeds']} successful rows); "
        f"{cfg['n_iterations']} iterations.\n\n"
        f"- Cells filled by seeding: **{res['n_filled_after_seeds']} / {n_cells}**\n"
        f"- Cells filled at end: **{len(grid)} / {n_cells}**\n"
        f"- Quality (signed forward speed) across the grid: max "
        f"{grid['quality'].max():+.4f}, mean {grid['quality'].mean():+.4f}, "
        f"min {grid['quality'].min():+.4f} m/s\n"
        f"- Evaluations: {len(gen)} ({res['n_seeds']} seeds + "
        f"{len(offspring)} offspring), failed: {n_failed}, of which "
        f"rejected as MuJoCo-unstable: {n_unstable}\n"
    )
