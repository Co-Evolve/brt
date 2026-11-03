"""Claim in box sentence 4: at a given energy, do large joint excursions buy
extra speed? For each energy row of search b with at least `min_cells` filled
cells, report the joint excursion of the fastest elite and the Spearman
correlation between excursion and speed across the row's elites.

    python scripts/excursion_by_energy.py               # reads output/search_b
    python scripts/excursion_by_energy.py --out DIR     # output root

Read-only on the run outputs: it reads grid_final.parquet, genealogy.parquet
and config.yaml of the search and writes only
<out>/analysis/excursion_by_energy.md.

Excursion is the elite's measured joint excursion (not the cell centre); edge
cells can hold values beyond the axis bounds. Spearman rho uses average ranks
for ties; p is a two-sided permutation p-value (shuffling speeds within the
row). The elites of a row are not independent samples, so p is descriptive.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse(search_dir: Path, cfg: dict) -> pd.DataFrame:
    grid = pd.read_parquet(search_dir / "grid_final.parquet")
    gen = pd.read_parquet(search_dir / "genealogy.parquet")
    run_cfg = yaml.safe_load((search_dir / "config.yaml").read_text())
    desc, n = run_cfg["descriptors"], run_cfg["grid_n"]
    elites = gen.loc[grid.solution_id, ["joint_excursion", "energy", "quality"]]
    elites = elites.reset_index(drop=True)
    assert np.array_equal(elites.quality.to_numpy(), grid.quality.to_numpy())
    g = pd.concat([grid.drop(columns="quality").reset_index(drop=True), elites], axis=1)
    edges = 10 ** np.linspace(desc["d2_min_log"], desc["d2_max_log"], n + 1)

    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for cy, r in g.groupby("cell_y_idx"):
        if len(r) < cfg["min_cells"]:
            continue
        x, y = r.joint_excursion.to_numpy(), r.quality.to_numpy()
        rho = spearman(x, y)
        perm = np.array(
            [spearman(x, rng.permutation(y)) for _ in range(cfg["n_permutations"])]
        )
        p = (np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (len(perm) + 1)
        best = r.loc[r.quality.idxmax()]
        rows.append(
            {
                "row": int(cy),
                "energy_lo_J": edges[cy],
                "energy_hi_J": edges[cy + 1],
                "n": len(r),
                "excursion_min": x.min(),
                "excursion_max": x.max(),
                "best_excursion": best.joint_excursion,
                "best_rank": int((x < best.joint_excursion).sum()) + 1,
                "best_speed": best.quality,
                "speed_at_max_excursion": r.loc[r.joint_excursion.idxmax()].quality,
                "rho": rho,
                "p": p,
            }
        )
    return pd.DataFrame(rows)


def to_markdown(t: pd.DataFrame, cfg: dict) -> str:
    fmt_p = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"  # noqa: E731
    lines = [
        "| row | energy (J) | n | excursion range (rad) | fastest elite: "
        "excursion (rad) | its rank by excursion | its speed (m/s) | speed at "
        "largest excursion (m/s) | Spearman ρ | p |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in t.itertuples():
        lines.append(
            f"| {r.row} | {r.energy_lo_J:.1f}–{r.energy_hi_J:.1f} | {r.n} | "
            f"{r.excursion_min:.2f}–{r.excursion_max:.2f} | {r.best_excursion:.2f} | "
            f"{r.best_rank} | {r.best_speed:.3f} | {r.speed_at_max_excursion:.3f} | "
            f"{r.rho:.2f}".replace("-", "−") + f" | {fmt_p(r.p)} |"
        )
    neg, pos = int((t.rho < 0).sum()), int((t.rho > 0).sum())
    sig = int(((t.p < 0.05) & (t.rho < 0)).sum())
    lines += [
        "",
        f"Rows analysed: {len(t)} (at least {cfg['min_cells']} filled cells). "
        f"ρ < 0 in {neg}, ρ > 0 in {pos}; ρ < 0 with p < 0.05 in {sig}. "
        f"Fastest elite in the row's lowest-excursion cell: "
        f"{int((t.best_rank == 1).sum())} rows. "
        f"p: two-sided permutation test, {cfg['n_permutations']:,} shuffles, "
        f"seed {cfg['seed']}.",
    ]
    return "\n".join(lines) + "\n"


def main(out: Path, cfg_path: Path) -> Path:
    cfg = yaml.safe_load(cfg_path.read_text())
    t = analyse(Path(out) / cfg["search"], cfg)
    md = to_markdown(t, cfg)
    dest = Path(out) / "analysis" / "excursion_by_energy.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md)
    print(md)
    print("written", dest)
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "output")
    ap.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "excursion_by_energy.yaml"
    )
    args = ap.parse_args()
    main(args.out, args.config)
