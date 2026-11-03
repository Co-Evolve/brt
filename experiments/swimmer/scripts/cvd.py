"""Colour-vision-deficiency check (Machado, Oliveira & Fernandes 2009, severity 1.0).
Usage: python3 cvd.py fig.png out_sheet.png"""
import sys
import numpy as np
from PIL import Image, ImageDraw
M = {
 "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                           [0.280085, 0.672501, 0.047413],
                           [-0.011820, 0.042940, 0.968881]]),
 "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                         [0.114503, 0.786281, 0.099216],
                         [-0.003882, -0.048116, 1.051998]]),
}
def to_lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
def simulate(rgb, kind):
    lin = to_lin(rgb)
    return to_srgb(lin @ M[kind].T)
def lab(rgb):
    lin = to_lin(np.asarray(rgb, float))
    X = lin @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                        [0.0193, 0.1192, 0.9505]]).T
    X = X / np.array([0.95047, 1.0, 1.08883])
    f = np.where(X > 0.008856, np.cbrt(X), 7.787 * X + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)
if __name__ == "__main__":
    im = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(float) / 255
    tiles = [Image.fromarray((im * 255).astype("uint8"))]
    for k in M:
        tiles.append(Image.fromarray((simulate(im, k) * 255).round().astype("uint8")))
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w, h * 3 + 60), "white")
    d = ImageDraw.Draw(sheet)
    for i, (t, name) in enumerate(zip(tiles, ["original", "deuteranopia", "protanopia"])):
        sheet.paste(t, (0, i * (h + 30)))
        d.text((10, i * (h + 30) + h + 4), name, fill="black")
    sheet.save(sys.argv[2])
    roles = {"morph": "#eed49f", "ctrl": "#1e66f5", "env": "#40a02b",
             "beh": "#c0142f", "alt": "#9ca0b0", "agar": "#e2f1df", "engine": "#ccd0da"}
    hexrgb = lambda c: np.array([int(c[i:i+2], 16) for i in (1, 3, 5)]) / 255
    for k in ["normal"] + list(M):
        cols = {n: (hexrgb(c) if k == "normal" else simulate(hexrgb(c), k)) for n, c in roles.items()}
        L = {n: lab(v) for n, v in cols.items()}
        pairs = [("beh", "morph"), ("beh", "env"), ("beh", "alt"), ("ctrl", "alt"),
                 ("env", "morph"), ("beh", "ctrl"), ("agar", "engine")]
        print(k.ljust(13), "  ".join(f"{a}-{b}:{np.linalg.norm(L[a]-L[b]):5.1f}" for a, b in pairs))
