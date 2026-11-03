"""Reproduce everything: the sweep, then both MAP-Elites searches, then the
box figure.

    python scripts/reproduce_all.py            # writes ./output/
    python scripts/reproduce_all.py --jobs 2   # run the two searches in parallel

The searches are independent given the sweep, so --jobs 2 gives the same
bits in about half the time. A SHA-256 manifest of every result file is
written to output/checksums.txt for run-to-run comparison.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from swimmer.runs import do_search, do_sweep, load_config, sha256  # noqa: E402

RESULT_FILES = [
    "sweep/data.parquet",
    "search_a/genealogy.parquet",
    "search_a/grid_final.parquet",
    "search_b/genealogy.parquet",
    "search_b/grid_final.parquet",
]


def _search(cfg_path: str, out: str) -> str:
    return str(do_search(load_config(Path(cfg_path)), Path(out)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "output")
    ap.add_argument("--configs", type=Path, default=ROOT / "configs")
    ap.add_argument("--jobs", type=int, default=1, choices=(1, 2))
    args = ap.parse_args()
    out = args.out.resolve()
    t0 = time.perf_counter()

    do_sweep(load_config(args.configs / "sweep.yaml"), out)

    searches = [str(args.configs / f"search_{k}.yaml") for k in ("a", "b")]
    if args.jobs == 2:
        with ProcessPoolExecutor(max_workers=2) as pool:
            list(pool.map(_search, searches, [str(out)] * 2))
    else:
        for s in searches:
            _search(s, str(out))

    import excursion_by_energy  # read-only analysis of search b

    excursion_by_energy.main(out, args.configs / "excursion_by_energy.yaml")

    lines = [f"{sha256(out / f)}  {f}" for f in RESULT_FILES]
    (out / "checksums.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import make_figure

    make_figure.main(out / "search_a", out / "search_b", out / "figure")
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
