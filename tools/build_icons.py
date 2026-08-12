#!/usr/bin/env python3
"""Generate the PWA icons into web/public/icons/.

    .venv/bin/python tools/build_icons.py

These are placeholders in the sense that no designer drew them, but they are not
throwaway: they use the app's own palette, and the maskable variant respects
Android's safe zone. Replace the leaf mark when real artwork exists; the sizes
and the safe-zone geometry should stay as they are.

Android will not offer "Add to Home Screen" without these -- the manifest
references all three, and a manifest pointing at a 404 fails installability
silently rather than loudly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "web" / "public" / "icons"

BG = "#0b1f16"  # --bg
ACCENT = "#4ade80"  # --accent

# A leaf on water: a pointed ellipse with a midrib, over two ripple lines.
# Drawn in a 512-unit viewBox and scaled, so one path serves every size.
LEAF = (
    "M256 108"
    "C330 150 372 220 372 286"
    "C372 350 320 396 256 396"
    "C192 396 140 350 140 286"
    "C140 220 182 150 256 108"
    "Z"
)
MIDRIB = "M256 132 L256 380"
RIPPLE_1 = "M108 424 C150 404 190 444 232 424 C274 404 314 444 356 424 C382 411 400 418 412 424"
RIPPLE_2 = "M108 462 C150 442 190 482 232 462 C274 442 314 482 356 462 C382 449 400 456 412 462"


def svg(size: int, maskable: bool) -> str:
    # Android crops maskable icons to an arbitrary shape and guarantees only the
    # centre 80% survives. Shrinking the artwork into that safe zone is the whole
    # difference between the two variants -- a full-bleed icon gets its edges
    # clipped on circular/squircle launchers.
    scale = 0.80 if maskable else 1.0
    offset = (1 - scale) * 256

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 512 512">
  <rect width="512" height="512" fill="{BG}"/>
  <g transform="translate({offset:.1f},{offset:.1f}) scale({scale})">
    <path d="{LEAF}" fill="{ACCENT}"/>
    <path d="{MIDRIB}" stroke="{BG}" stroke-width="16" stroke-linecap="round" fill="none"/>
    <path d="{RIPPLE_1}" stroke="{ACCENT}" stroke-width="18" stroke-linecap="round" fill="none" opacity="0.85"/>
    <path d="{RIPPLE_2}" stroke="{ACCENT}" stroke-width="14" stroke-linecap="round" fill="none" opacity="0.5"/>
  </g>
</svg>"""


def main() -> int:
    try:
        import cairosvg
    except ImportError:
        print(
            "cairosvg is required:  .venv/bin/python -m pip install cairosvg",
            file=sys.stderr,
        )
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        ("icon-192.png", 192, False),
        ("icon-512.png", 512, False),
        ("icon-512-maskable.png", 512, True),
        # Not in the manifest, but iOS looks for a link tag pointing at one and
        # falls back to a screenshot of the page if it is absent.
        ("apple-touch-icon.png", 180, True),
    ]

    for name, size, maskable in targets:
        out = OUT_DIR / name
        cairosvg.svg2png(
            bytestring=svg(size, maskable).encode("utf-8"),
            write_to=str(out),
            output_width=size,
            output_height=size,
        )
        print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1024:.1f} KB")

    # Keep the source mark alongside the raster output so it can be edited.
    (OUT_DIR / "icon.svg").write_text(svg(512, False), encoding="utf-8")
    print(f"{(OUT_DIR / 'icon.svg').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
