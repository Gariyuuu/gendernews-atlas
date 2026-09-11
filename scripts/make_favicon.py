#!/usr/bin/env python
"""make favicon -- the site mark, generated from one geometry so SVG, PNG and ICO agree.

The mark: an ink tile carrying the masthead's double rule, and below it an opening
quotation mark in the chart vermilion (quotation is the atlas's firmest finding). Drawn as
geometry, not type: favicons cannot load web fonts, and a text glyph renders differently
on every OS.

Outputs (committed; re-run only when the mark changes)
  site/assets/favicon.svg           browsers that take SVG icons
  site/assets/favicon.ico           16/32/48, copied to the site root by site/build.py
  site/assets/favicon-32.png
  site/assets/apple-touch-icon.png  180 px, full-bleed (iOS rounds it itself)
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "assets"
PAL = json.loads((ROOT / "config" / "palette.json").read_text())["light"]
INK, PAPER, MARK = PAL["primary"], PAL["surface"], PAL["women"]

S = 64                      # design grid
RULES = [(12, 12, 40, 4.0), (12, 20, 40, 2.0)]            # x, y, w, h (y down); on the 16 px grid (4 units = 1 px)
BALLS = [(23.0, 45.0), (41.0, 45.0)]                       # quote-mark ball centres
R = 7.0


def tail(cx: float, cy: float) -> list[tuple[str, list[tuple[float, float]]]]:
    """The rising, right-curving tail of one '6'-shaped half of an opening quote."""
    return [("M", [(cx - R, cy)]),
            ("C", [(cx - R, cy - 10), (cx - 2, cy - 16), (cx + 6, cy - 18)]),
            ("L", [(cx + 6.5, cy - 14.4)]),
            ("C", [(cx + 1.5, cy - 12.5), (cx - 1, cy - 9), (cx + 0.5, cy - 6.6)]),
            ("Z", [])]


def svg(rx: float = 13) -> str:
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}">',
             f'<rect width="{S}" height="{S}" rx="{rx}" fill="{INK}"/>']
    parts += [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{min(h / 2, 1)}" fill="{PAPER}"/>'
              for x, y, w, h in RULES]
    for cx, cy in BALLS:
        d = " ".join(op + " ".join(f"{px:g},{py:g}" for px, py in pts) for op, pts in tail(cx, cy))
        parts.append(f'<path d="{d}" fill="{MARK}"/><circle cx="{cx:g}" cy="{cy:g}" r="{R:g}" fill="{MARK}"/>')
    return "".join(parts) + "</svg>\n"


def png(px: int, rx: float) -> Image.Image:
    fig = plt.figure(figsize=(1, 1), dpi=px)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, S)
    ax.set_ylim(S, 0)                       # y down, as in the SVG
    ax.axis("off")
    if rx:
        ax.add_patch(FancyBboxPatch((0, 0), S, S, boxstyle=f"round,pad=0,rounding_size={rx}", fc=INK, ec="none"))
    else:
        ax.add_patch(Rectangle((0, 0), S, S, fc=INK, ec="none"))
    for x, y, w, h in RULES:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={min(h / 2, 1)}",
                                    fc=PAPER, ec="none"))
    codes = {"M": MPath.MOVETO, "L": MPath.LINETO, "C": MPath.CURVE4}
    for cx, cy in BALLS:
        verts, kinds = [], []
        for op, pts in tail(cx, cy):
            if op == "Z":
                verts.append(verts[0])
                kinds.append(MPath.CLOSEPOLY)
            else:
                verts += pts
                kinds += [codes[op]] * len(pts)
        ax.add_patch(PathPatch(MPath(verts, kinds), fc=MARK, ec="none"))
        ax.add_patch(Circle((cx, cy), R, fc=MARK, ec="none"))
    buf = io.BytesIO()
    fig.savefig(buf, dpi=px, transparent=True)
    plt.close(fig)
    return Image.open(buf).convert("RGBA").resize((px, px), Image.LANCZOS)


def main() -> int:
    (OUT / "favicon.svg").write_text(svg())
    png(32, 13).save(OUT / "favicon-32.png")
    png(180, 0).save(OUT / "apple-touch-icon.png")
    png(256, 13).save(OUT / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png")) + [OUT / "favicon.svg", OUT / "favicon.ico"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
