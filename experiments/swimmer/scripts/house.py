"""House style for the BioSim review figures.

Palette and drawing helpers taken from the approved controller figure
(figures/make_controller_figure.py), wrapped in a small SVG canvas class so
every figure script can reuse them. 1800 user units = 170 mm.

Conventions (see the project's figure style guide):
- text stays live (Helvetica), in slate ink, never pure black;
- arrowheads are explicit filled paths, never SVG markers (Affinity drops them);
- every panel is a named <g id="panel-..."> group;
- colour roles: morphology yellow, controller blue, environment green,
  behaviour and everything derived from it crimson, neutral grey for
  unchosen alternatives, physics engine as a neutral slate card.
"""
import base64
import io
import math


def hx(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def mix(c1, c2, t):
    a, b = hx(c1), hx(c2)
    return "#" + "".join(f"{round(a[i] + (b[i] - a[i]) * t):02x}"
                         for i in range(3))


W_ = "#ffffff"
INK = "#4c4f69"
CTRL = "#1e66f5"
ENV = "#40a02b"
BEH = "#c0142f"
MORPH = "#eed49f"

P = dict(
    ink=INK, sub="#5c5f77", muted="#7c7f93", faint="#bcc0cc",
    card_top="#dce0e8", card_bot="#eff1f5",
    ctrl=CTRL, ctrl_d=mix(CTRL, INK, 0.25), glow="#7287fd",
    ctrl_l=mix(CTRL, W_, 0.86),
    morph=MORPH, morph_d="#b8730f", morph_l=mix("#df8e1d", W_, 0.62),
    morph_s=mix(MORPH, "#b8730f", 0.35),      # shaded side of a yellow solid
    morph_h=mix(MORPH, W_, 0.45),             # lit side of a yellow solid
    outline=INK,
    env=ENV, env_d=mix(ENV, INK, 0.3),
    agar=mix(ENV, W_, 0.84), agar_c=mix(ENV, W_, 0.93),
    rim=mix(ENV, W_, 0.42), rim2=mix(ENV, W_, 0.68),
    groove_d=mix(ENV, W_, 0.62), groove_l=mix(ENV, W_, 0.94),
    env_text=ENV,
    beh=BEH, beh_l=mix(BEH, W_, 0.5), beh_xl=mix(BEH, W_, 0.88),
    alt="#9ca0b0", pill=INK,
    engine="#ccd0da", engine_d="#acb0be",     # Catppuccin surface0 / surface2
)

FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"


class SVG:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.out = []
        self.defs = []
        self.add(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'xmlns:xlink="http://www.w3.org/1999/xlink" width="170mm" '
                 f'height="{170 * H / W:.2f}mm" viewBox="0 0 {W} {H}" '
                 f'font-family="{FONT}">')
        self.add("@@DEFS@@")
        self.def_('<linearGradient id="card" x1="0" y1="0" x2="0" y2="1">'
                  f'<stop offset="0" stop-color="{P["card_top"]}"/>'
                  f'<stop offset="1" stop-color="{P["card_bot"]}"/>'
                  '</linearGradient>')
        self.rect(0, 0, W, H, "white", rx=0, extra='id="background"')

    # -- plumbing ------------------------------------------------------------
    def add(self, s):
        self.out.append(s)

    def def_(self, s):
        self.defs.append(s)

    def group(self, gid, extra=""):
        self.add(f'<g id="{gid}"{(" " + extra) if extra else ""}>')

    def end(self):
        self.add("</g>")

    def save(self, path):
        body = "\n".join(self.out + ["</svg>"])
        body = body.replace("@@DEFS@@", "<defs>" + "".join(self.defs) +
                            "</defs>", 1)
        with open(path, "w") as f:
            f.write(body)
        print("written", path)

    # -- primitives ----------------------------------------------------------
    def text(self, x, y, s, size=24, anchor="start", fill=None,
             weight="normal", style=None, extra=""):
        st = f' font-style="{style}"' if style else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
                 f'text-anchor="{anchor}" fill="{fill or P["ink"]}" '
                 f'font-weight="{weight}"{st}{extra}>{s}</text>')

    def head(self, x0, y0, x1, y1, colour, size=15, width=11):
        d = math.dist((x0, y0), (x1, y1))
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        nx, ny = -uy, ux
        bx, by = x1 - ux * size, y1 - uy * size
        self.add(f'<path d="M {x1:.1f} {y1:.1f} L {bx + nx * width / 2:.1f} '
                 f'{by + ny * width / 2:.1f} L {bx - nx * width / 2:.1f} '
                 f'{by - ny * width / 2:.1f} Z" fill="{colour}"/>')

    def line(self, x1, y1, x2, y2, stroke=None, sw=3.2, arrow=False,
             dash=None, opacity=None, cap="round", hs=15):
        stroke = stroke or P["ink"]
        da = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' opacity="{opacity}"' if opacity is not None else ""
        if arrow:
            d = math.dist((x1, y1), (x2, y2))
            ex = x2 - (x2 - x1) / d * (hs - 3)
            ey = y2 - (y2 - y1) / d * (hs - 3)
        else:
            ex, ey = x2, y2
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{ex:.1f}" '
                 f'y2="{ey:.1f}" stroke="{stroke}" stroke-width="{sw}" '
                 f'stroke-linecap="{cap}"{da}{op}/>')
        if arrow:
            self.head(x1, y1, x2, y2, stroke, size=hs, width=hs * 0.75)

    def path(self, d, stroke=None, sw=3.2, fill="none", dash=None,
             opacity=None, tip=None, hs=15, extra="", join="round",
             cap="round"):
        stroke = stroke or P["ink"]
        da = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}" stroke-linecap="{cap}" '
                 f'stroke-linejoin="{join}"{da}{op}{extra}/>')
        if tip:
            self.head(*tip, stroke, size=hs, width=hs * 0.75)

    def polyline(self, pts, stroke, sw, opacity=None, dash=None, cap="round"):
        op = f' opacity="{opacity}"' if opacity is not None else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<polyline points="'
                 f'{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" '
                 f'fill="none" stroke="{stroke}" stroke-width="{sw}" '
                 f'stroke-linecap="{cap}" stroke-linejoin="round"{op}{da}/>')

    def polygon(self, pts, fill, stroke="none", sw=0, opacity=None,
                join="round"):
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<polygon points="'
                 f'{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
                 f'stroke-linejoin="{join}"{op}/>')

    def rect(self, x, y, w, h, fill, rx=14, stroke="none", sw=0, dash=None,
             extra="", opacity=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                 f'height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}"{da}{op} {extra}/>')

    def circle(self, cx, cy, r, fill, stroke="none", sw=0, dash=None,
               opacity=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
                 f'{da}{op}/>')

    def ellipse(self, cx, cy, rx, ry, fill, stroke="none", sw=0, rot=0,
                opacity=None):
        op = f' opacity="{opacity}"' if opacity is not None else ""
        tr = (f' transform="rotate({rot:.1f} {cx:.1f} {cy:.1f})"'
              if rot else "")
        self.add(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" '
                 f'ry="{ry:.1f}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}"{tr}{op}/>')

    # -- house components ----------------------------------------------------
    def pill(self, cx, cy, label, size=25, h=46, fill=None, fg="white",
             w=None, nchar=None):
        n = nchar if nchar is not None else len(label)
        w = w or 0.54 * size * n + 46
        self.rect(cx - w / 2, cy - h / 2, w, h, fill or P["pill"], rx=h / 2)
        self.text(cx, cy + size * 0.36, label, size=size, anchor="middle",
                  fill=fg)
        return w

    def chip(self, cx, cy, label, fill, fg=None, stroke=None, size=21, h=34,
             w=None, sw=2):
        w = w or 0.54 * size * len(label) + 30
        self.rect(cx - w / 2, cy - h / 2, w, h, fill, rx=h / 2,
                  stroke=stroke or "none", sw=sw if stroke else 0)
        self.text(cx, cy + size * 0.36, label, size=size, anchor="middle",
                  fill=fg or P["ink"])
        return w

    def card(self, x, y, w, h, letter=None, title=None, title_x=None,
             title_chars=None):
        self.rect(x, y, w, h, "url(#card)", rx=18)
        if letter:
            self.text(x + 22, y + 62, letter, size=32, weight="bold")
        if title:
            self.pill(title_x if title_x is not None else x + w / 2, y, title,
                      nchar=title_chars)

    def gap_label(self, x1, x2, y, label, colour, size=22, label_x=None,
                  sw=3.2, dash=None, nchar=None):
        lx = label_x if label_x is not None else (x1 + x2) / 2
        n = nchar if nchar is not None else len(label)
        half = 0.3 * size * n + 12
        lo, hi = sorted([x1, x2])
        self.line(lo, y, lx - half, y, stroke=colour, sw=sw, dash=dash)
        self.line(lx + half, y, hi, y, stroke=colour, sw=sw, dash=dash)
        self.text(lx, y + size * 0.36, label, size=size, anchor="middle",
                  fill=colour)

    def label_line(self, x, y, tx, ty, label_lines, anchor="start", size=21,
                   fill=None, dot=True, lead=None):
        """Callout: a thin ink leader from (x, y) on the object to the text
        anchored at (tx, ty); first text line is level with ty."""
        fill = fill or P["sub"]
        lead = lead or P["ink"]
        gap = 8 if anchor == "start" else -8 if anchor == "end" else 0
        self.line(x, y, tx - gap, ty - size * 0.32, stroke=lead, sw=1.6)
        if dot:
            self.circle(x, y, 4, lead)
        for i, s in enumerate(label_lines):
            self.text(tx, ty + i * size * 1.18, s, size=size, anchor=anchor,
                      fill=fill)

    def image(self, x, y, w, h, png_bytes, mime="image/png", clip_rx=None,
              iid=None, preserve="xMidYMid meet"):
        b64 = base64.b64encode(png_bytes).decode()
        cp = ""
        if clip_rx is not None:
            cid = f"clip-{iid or len(self.defs)}"
            self.def_(f'<clipPath id="{cid}"><rect x="{x:.1f}" y="{y:.1f}" '
                      f'width="{w:.1f}" height="{h:.1f}" rx="{clip_rx}"/>'
                      f'</clipPath>')
            cp = f' clip-path="url(#{cid})"'
        idattr = f' id="{iid}"' if iid else ""
        self.add(f'<image{idattr} x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                 f'height="{h:.1f}" preserveAspectRatio="{preserve}"{cp} '
                 f'xlink:href="data:{mime};base64,{b64}"/>')

    def dish(self, cx, cy, r, gid):
        self.def_(f'<radialGradient id="{gid}" cx="0.45" cy="0.4" r="0.65">'
                  f'<stop offset="0" stop-color="{P["agar_c"]}"/>'
                  f'<stop offset="1" stop-color="{P["agar"]}"/>'
                  '</radialGradient>')
        self.circle(cx, cy, r, f"url(#{gid})", P["rim"], 5)
        self.circle(cx, cy, r - 10, "none", P["rim2"], 2)

    def gear(self, cx, cy, r, fill=None, stroke=None, teeth=8, sw=2.2):
        fill = fill or "white"
        stroke = stroke or P["ink"]
        pts = []
        for i in range(teeth * 4):
            a = 2 * math.pi * i / (teeth * 4)
            rr = r if (i % 4) in (0, 1) else r * 0.78
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        self.polygon(pts, fill, stroke, sw)
        self.circle(cx, cy, r * 0.32, P["engine"], stroke, sw)


# --------------------------------------------------------------------------
# Raster helpers
# --------------------------------------------------------------------------
def png_bytes(im, fmt="PNG", quality=92):
    buf = io.BytesIO()
    if fmt == "JPEG":
        im.convert("RGB").save(buf, "JPEG", quality=quality)
    else:
        im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def recolour(im, src_hue, dst_rgb, hue_tol=28, min_sat=0.12,
             strength=1.0):
    """Map pixels whose hue is near src_hue (degrees) onto the hue and
    saturation of dst_rgb, keeping their lightness, so a rendered teal body
    becomes the morphology yellow with its shading intact. Pixels of other
    hues (red contact highlights, grey background) are left alone."""
    import colorsys
    import numpy as np
    arr = np.asarray(im.convert("RGBA")).astype(float) / 255.0
    rgb, a = arr[..., :3], arr[..., 3:]
    mx, mn = rgb.max(-1), rgb.min(-1)
    l = (mx + mn) / 2
    d = mx - mn
    s = np.where(d == 0, 0, d / (1 - np.abs(2 * l - 1) + 1e-9))
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.zeros_like(l)
    m = d > 0
    rc = np.where(m, (mx - r) / (d + 1e-9), 0)
    gc = np.where(m, (mx - g) / (d + 1e-9), 0)
    bc = np.where(m, (mx - b) / (d + 1e-9), 0)
    h = np.where(r == mx, bc - gc, np.where(g == mx, 2 + rc - bc, 4 + gc - rc))
    h = (h / 6.0) % 1.0 * 360
    dh = np.abs((h - src_hue + 180) % 360 - 180)
    w = np.clip(1 - (dh - hue_tol * 0.6) / (hue_tol * 0.4), 0, 1)
    w = w * np.clip((s - min_sat) / 0.08, 0, 1) * strength
    th, tl, ts = colorsys.rgb_to_hls(*[c / 255 for c in hx(dst_rgb)])
    # keep relative lightness: map the body's lightness range onto a range
    # centred on the target lightness
    lm = l[w > 0.5].mean() if (w > 0.5).any() else 0.5
    nl = np.clip(tl + (l - lm) * 1.05, 0, 1)
    ns = np.full_like(l, ts) * np.clip(1 - np.abs(nl - tl) * 0.6, 0.4, 1)
    # hls -> rgb, vectorised
    q = np.where(nl < 0.5, nl * (1 + ns), nl + ns - nl * ns)
    p = 2 * nl - q

    def h2c(t):
        t = t % 1.0
        return np.where(t < 1 / 6, p + (q - p) * 6 * t,
                        np.where(t < 0.5, q,
                                 np.where(t < 2 / 3, p + (q - p) * (2 / 3 - t)
                                          * 6, p)))
    new = np.stack([h2c(th + 1 / 3), h2c(th), h2c(th - 1 / 3)], -1)
    out = rgb * (1 - w[..., None]) + new * w[..., None]
    from PIL import Image
    return Image.fromarray((np.concatenate([out, a], -1) * 255)
                           .round().astype("uint8"), "RGBA")


# --------------------------------------------------------------------------
# Text measurement (Helvetica is rendered as TeX Gyre Heros here)
# --------------------------------------------------------------------------
_FONTS = {}


def text_width(s, size):
    import re
    from PIL import ImageFont
    import subprocess
    if "path" not in _FONTS:
        _FONTS["path"] = subprocess.run(
            ["fc-match", "-f", "%{file}", "Helvetica"], capture_output=True,
            text=True).stdout.strip()
    key = round(size * 4)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(_FONTS["path"], key)
    plain = re.sub(r"<[^>]+>", "", s)
    return _FONTS[key].getlength(plain) / 4


def plate(c, x, y, lines, size=21, anchor="start", fill=None, pad=7,
          bg="white", opacity=0.88, lh=1.22, weight="normal", style=None):
    """Text on a soft white plate, for labels that sit on top of shaded
    regions (keeps them legible in print and under colour blindness)."""
    w = max(text_width(s, size) for s in lines)
    hgt = size * (1 + lh * (len(lines) - 1))
    x0 = x if anchor == "start" else x - w if anchor == "end" else x - w / 2
    c.rect(x0 - pad, y - size * 0.86 - pad * 0.6, w + 2 * pad,
           hgt + pad * 1.3, bg, rx=8, opacity=opacity)
    for i, s in enumerate(lines):
        c.text(x, y + i * size * lh, s, size=size, anchor=anchor,
               fill=fill or P["ink"], weight=weight, style=style)
    return w
