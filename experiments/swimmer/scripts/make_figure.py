"""The box figure: two MAP-Elites grids side by side, house style, SVG.

    python scripts/make_figure.py                     # reads/writes ./output/
    python scripts/make_figure.py --a DIR --b DIR --out DIR

Left = search a (log10 speed x log10 energy), right = search b (joint
excursion x log10 energy). Reads only grid_final.parquet and config.yaml of
each search; no simulation. 1800 SVG units = 170 mm.

Colour: signed forward speed on one shared sequential scale, a single crimson
hue (the house behaviour colour) whose OKLab lightness falls linearly from
light to dark, so equal speed steps are equal perceived steps. The light end
stays well off white: white is reserved for cells the search never reached.
Backward-swimming elites (speed < 0) also get a thick dashed ink outline.
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house import SVG, P, plate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

VMIN, VMAX = -0.10, 0.70  # m/s; set in main() from the data, rounded out to 0.05
L_LIGHT, L_DARK = 0.94, 0.32  # OKLab lightness at VMIN and VMAX
NAME = "box_descriptor_choice"

# --------------------------------------------------------------------------
# Colour: OKLab <-> sRGB (Ottosson 2020)
# --------------------------------------------------------------------------


def _lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _gam(c):
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def srgb_to_oklab(hexc):
    r, g, b = (_lin(int(hexc[i : i + 2], 16) / 255) for i in (1, 3, 5))
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def oklab_to_lin(L, a, b):
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )


_L0, _A0, _B0 = srgb_to_oklab(P["beh"])
HUE = math.atan2(_B0, _A0)  # crimson hue angle of the house behaviour colour


def ramp(t):
    """t in [0, 1] -> hex. Lightness linear in t, constant hue, chroma as
    high as sRGB allows (capped by a smooth envelope)."""
    L = L_LIGHT + (L_DARK - L_LIGHT) * t
    cmax = 0.05 + 0.17 * math.sin(math.pi * min(1.0, 0.25 + 0.75 * t)) ** 0.7
    lo, hi = 0.0, cmax
    for _ in range(40):  # largest in-gamut chroma <= cmax
        c = (lo + hi) / 2
        rgb = oklab_to_lin(L, c * math.cos(HUE), c * math.sin(HUE))
        lo, hi = (c, hi) if all(-1e-6 <= v <= 1 + 1e-6 for v in rgb) else (lo, c)
    rgb = oklab_to_lin(L, lo * math.cos(HUE), lo * math.sin(HUE))
    return "#" + "".join(f"{round(255 * _gam(min(1, max(0, v)))):02x}" for v in rgb)


def colour(speed):
    return ramp(min(1.0, max(0.0, (speed - VMIN) / (VMAX - VMIN))))


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
W, H = 1800, 1005
CARD_Y, CARD_W, CARD_H = 40, 868, 820
CARD_X = (21, 911)
PLOT = 600  # plot side; 20 cells of 30 units
PX, PY = 172, 112  # plot origin offset inside the card
TICK = 18  # tick-label size
SUP = lambda s: f'<tspan dy="-9" font-size="14">{s}</tspan>'  # noqa: E731


def load(run_dir):
    run_dir = Path(run_dir)
    grid = pd.read_parquet(run_dir / "grid_final.parquet")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    return grid, cfg


def draw_grid(c, x0, y0, grid, n):
    cw = PLOT / n
    c.rect(x0, y0, PLOT, PLOT, "white", rx=0)
    for k in range(n + 1):  # faint lattice so empty cells read as cells
        c.line(
            x0 + k * cw,
            y0,
            x0 + k * cw,
            y0 + PLOT,
            stroke=P["faint"],
            sw=0.8,
            opacity=0.55,
            cap="butt",
        )
        c.line(
            x0,
            y0 + k * cw,
            x0 + PLOT,
            y0 + k * cw,
            stroke=P["faint"],
            sw=0.8,
            opacity=0.55,
            cap="butt",
        )
    cells = {}
    for r in grid.itertuples():
        cx, cy = int(r.cell_x_idx), int(r.cell_y_idx)
        x, y = x0 + cx * cw, y0 + PLOT - (cy + 1) * cw
        c.rect(x + 1, y + 1, cw - 2, cw - 2, colour(r.quality), rx=2)
        cells[(cx, cy)] = (x, y, r.quality)
    for (cx, cy), (x, y, q) in cells.items():  # outlines on top
        if q < 0:
            c.rect(
                x + 2.5,
                y + 2.5,
                cw - 5,
                cw - 5,
                "none",
                rx=2,
                stroke=P["ink"],
                sw=3.4,
                dash="5.5 3.5",
            )
    return cells, cw


def axes(c, x0, y0, xticks, yticks, xlabel, ylabel, xr, yr):
    """xticks/yticks: (value, label) in descriptor units; xr/yr: (lo, hi)."""
    x1, y1 = x0 + PLOT, y0 + PLOT
    c.line(x0, y1, x1 + 22, y1, stroke=P["ink"], sw=2.4, arrow=True, hs=13)
    c.line(x0, y1, x0, y0 - 22, stroke=P["ink"], sw=2.4, arrow=True, hs=13)
    for v, lab in xticks:
        x = x0 + (v - xr[0]) / (xr[1] - xr[0]) * PLOT
        c.line(x, y1, x, y1 + 8, stroke=P["ink"], sw=1.8, cap="butt")
        c.text(x, y1 + 32, lab, size=TICK, anchor="middle", fill=P["sub"])
    for v, lab in yticks:
        y = y1 - (v - yr[0]) / (yr[1] - yr[0]) * PLOT
        c.line(x0 - 8, y, x0, y, stroke=P["ink"], sw=1.8, cap="butt")
        c.text(x0 - 14, y + 6, lab, size=TICK, anchor="end", fill=P["sub"])
    c.text(x0 + PLOT / 2, y1 + 70, xlabel, size=22, anchor="middle", fill=P["sub"])
    ly = y0 + PLOT / 2
    c.text(
        x0 - 92,
        ly,
        ylabel,
        size=22,
        anchor="middle",
        fill=P["sub"],
        extra=f' transform="rotate(-90 {x0 - 92:.1f} {ly:.1f})"',
    )


def main(dir_a, dir_b, out_dir):
    ga, ca = load(dir_a)
    gb, cb = load(dir_b)
    n = ca["grid_n"]
    assert cb["grid_n"] == n
    da, db = ca["descriptors"], cb["descriptors"]
    lo, hi = min(ga.quality.min(), gb.quality.min()), max(
        ga.quality.max(), gb.quality.max()
    )
    global VMIN, VMAX  # one shared scale for both panels, from the data
    VMIN = math.floor(min(lo, 0.0) / 0.05) * 0.05
    VMAX = math.ceil(hi / 0.05) * 0.05

    c = SVG(W, H)

    # ---------------- panel a ----------------
    X = CARD_X[0]
    c.group("panel-a-speed-energy")
    c.card(X, CARD_Y, CARD_W, CARD_H, letter="a", title="descriptors: speed × energy")
    x0, y0 = X + PX, CARD_Y + PY
    cells_a, cw = draw_grid(c, x0, y0, ga, n)
    xr = (da["speed_min_log"], da["speed_max_log"])
    yr = (da["energy_min_log"], da["energy_max_log"])
    axes(
        c,
        x0,
        y0,
        [
            (-4, "10" + SUP("−4")),
            (-3, "10" + SUP("−3")),
            (-2, "10" + SUP("−2")),
            (-1, "10" + SUP("−1")),
        ],
        [(0, "1"), (1, "10"), (2, "100")],
        "head-first speed (m/s, log scale)",
        "energy (J, log scale)",
        xr,
        yr,
    )
    c.text(
        X + CARD_W - 26,
        CARD_Y + 62,
        f"{len(ga)} of {n * n} cells filled",
        size=21,
        anchor="end",
        fill=P["sub"],
    )
    # clip-column callout
    # the three top rows of columns 1-6 are empty: put the note there
    tx, ty = x0 + cw + 12, y0 + 0.5 * cw + 6
    c.line(x0 + cw - 2, ty - 6, tx - 5, ty - 6, stroke=P["ink"], sw=1.6)
    for i, s in enumerate(
        [
            "speeds below 10" + SUP("−4") + '<tspan dy="9"> m/s</tspan>',
            "(tail-first included)",
            "are clipped into this column",
        ]
    ):
        c.text(tx, ty + i * cw, s, size=18, fill=P["sub"])
    c.end()

    # ---------------- panel b ----------------
    X = CARD_X[1]
    c.group("panel-b-excursion-energy")
    c.card(
        X,
        CARD_Y,
        CARD_W,
        CARD_H,
        letter="b",
        title="descriptors: joint excursion × energy",
    )
    x0, y0 = X + PX, CARD_Y + PY
    cells_b, cw = draw_grid(c, x0, y0, gb, n)
    xr = (db["d1_min"], db["d1_max"])
    yr = (db["d2_min_log"], db["d2_max_log"])
    axes(
        c,
        x0,
        y0,
        [(v, f"{v:.1f}") for v in (0.5, 1.0, 1.5, 2.0)],
        [(math.log10(v), str(v)) for v in (20, 50, 100, 200, 400)],
        "joint excursion (rad)",
        "energy (J, log scale)",
        xr,
        yr,
    )
    c.text(
        X + CARD_W - 26,
        CARD_Y + 62,
        f"{len(gb)} of {n * n} cells filled",
        size=21,
        anchor="end",
        fill=P["sub"],
    )
    # high-energy, low-excursion corner: 4 x 4 cells
    k = 4
    corner = gb[(gb.cell_x_idx < k) & (gb.cell_y_idx >= n - k)]
    c.rect(x0, y0, k * cw, k * cw, "none", rx=0, stroke=P["ink"], sw=2, dash="3 4")
    c.text(
        x0 + 20,
        y0 - 14,
        f"high energy, low excursion: {len(corner)} of {k * k} cells reached",
        size=19,
        fill=P["sub"],
    )
    # backward elites: speed on a plate just left of the cell
    for x, y, q in cells_b.values():
        if q < 0:
            lab = f"−{abs(q):.2f}" if abs(q) >= 0.01 else f"−{abs(q):.4f}"
            plate(
                c,
                x - 10,
                y + cw / 2 + 6,
                [lab + " m/s"],
                size=17,
                anchor="end",
                weight="bold",
                opacity=0.94,
            )
    c.end()

    # ---------------- shared legend ----------------
    c.group("legend-speed-scale")
    ly = CARD_Y + CARD_H + 48
    bx0, bw, bh = 560, 520, 22
    steps = 120
    for i in range(steps):
        t = i / (steps - 1)
        c.rect(bx0 + i * bw / steps, ly, bw / steps + 0.6, bh, ramp(t), rx=0)
    c.rect(bx0, ly, bw, bh, "none", rx=0, stroke=P["faint"], sw=1)
    ticks = [round(0.1 * k, 1) for k in range(-9, 10)]
    for v in [t for t in ticks if VMIN - 1e-9 <= t <= VMAX + 1e-9]:
        x = bx0 + (v - VMIN) / (VMAX - VMIN) * bw
        c.line(x, ly + bh, x, ly + bh + 7, stroke=P["ink"], sw=1.6, cap="butt")
        lab = "0" if v == 0 else f"{v:+.1f}".replace("-", "−")
        c.text(x, ly + bh + 28, lab, size=TICK, anchor="middle", fill=P["sub"])
    c.text(
        bx0 - 16,
        ly + 18,
        "head-first speed (m/s)",
        size=21,
        anchor="end",
        fill=P["sub"],
    )
    sx = bx0 + bw + 50
    c.rect(sx, ly - 1, 24, 24, colour(VMIN), rx=2)
    c.rect(
        sx + 1.5,
        ly + 0.5,
        21,
        21,
        "none",
        rx=2,
        stroke=P["ink"],
        sw=3.4,
        dash="5.5 3.5",
    )
    c.text(sx + 36, ly + 18, "swims tail first", size=21, fill=P["sub"])
    sx2 = sx + 240
    c.rect(sx2, ly - 1, 24, 24, "white", rx=2, stroke=P["faint"], sw=1.6)
    c.text(sx2 + 36, ly + 18, "not reached by this search", size=21, fill=P["sub"])
    c.end()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{NAME}.svg"
    c.save(path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, default=ROOT / "output" / "search_a")
    ap.add_argument("--b", type=Path, default=ROOT / "output" / "search_b")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "figure")
    args = ap.parse_args()
    main(args.a, args.b, args.out)
