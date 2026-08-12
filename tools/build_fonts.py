#!/usr/bin/env python3
"""Build the bundled Noto Sans Tamil web fonts.

Downloads the upstream variable font, pins it to the two weights the app uses,
subsets it, and writes WOFF2 into web/public/fonts/.

Run after `pip install "fonttools[woff]" brotli`:

    .venv/bin/python tools/build_fonts.py

Why this exists rather than a CDN link: the app must work offline, and Android
coverage for Tamil is inconsistent -- the failure mode is tofu, empty boxes
where the text should be.

SUBSETTING IS DELIBERATELY CONSERVATIVE. Tamil is an abugida: vowel signs and
the pulli combine with consonants to form clusters, and the shaping engine needs
both the combining marks and the GSUB/GPOS tables to assemble them. Subsetting
by observed character frequency -- the usual web-font tactic -- silently strips
marks that appear only in words nobody happened to type during testing, and
produces broken clusters in the field. So the whole Tamil block is kept, plus
the layout tables, and we accept the extra bytes.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "web" / "public" / "fonts"

UPSTREAM = (
    "https://github.com/google/fonts/raw/main/ofl/notosanstamil/"
    "NotoSansTamil%5Bwdth,wght%5D.ttf"
)

# Unicode ranges to retain. Bare hex, no "U+" prefix -- fontTools' --unicodes
# parser splits ranges on "-" and mis-parses the prefixed form.
#
# 0B80-0BFF is the Tamil block in full, including the combining vowel signs and
# virama that clusters are built from. The Latin/punctuation ranges are kept
# because the UI mixes scripts: numbers, the locale switch, and species
# binomials all render in this font.
UNICODES = ",".join(
    [
        "0000-007F",  # Basic Latin: digits, ASCII punctuation
        "00A0-00FF",  # Latin-1: non-breaking space, degree sign (coords)
        "0B80-0BFF",  # Tamil, complete
        "200B-200D",  # ZWSP/ZWNJ/ZWJ: cluster-breaking controls
        "2010-2027",  # dashes, quotes, ellipsis
        "20B9",  # Indian rupee
        "25CC",  # dotted circle: renders isolated combining marks
    ]
)

# wght is the only axis the CSS uses; wdth is pinned to its default. Instancing
# rather than shipping the variable font halves the payload, and the app only
# ever asks for 400 and 700.
WEIGHTS = {"Regular": 400, "Bold": 700}


def build(src: Path, name: str, weight: int) -> Path:
    out = OUT_DIR / f"NotoSansTamil-{name}.woff2"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fontTools.varLib.instancer",
            str(src),
            f"wght={weight}",
            "--output",
            str(OUT_DIR / f".tmp-{name}.ttf"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fontTools.subset",
            str(OUT_DIR / f".tmp-{name}.ttf"),
            f"--unicodes={UNICODES}",
            # Retain the shaping tables. Without GSUB/GPOS the glyphs are all
            # present but never combine, so clusters render as loose sequences
            # of base letters and floating marks.
            "--layout-features=*",
            "--flavor=woff2",
            f"--output-file={out}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (OUT_DIR / f".tmp-{name}.ttf").unlink()
    return out


def main() -> int:
    try:
        return _run()
    except subprocess.CalledProcessError as exc:
        # Surface the tool's own diagnostic; a bare traceback here says only
        # that fontTools exited non-zero, not which argument it choked on.
        print(f"{' '.join(exc.cmd)}\n{exc.stderr or exc.stdout}", file=sys.stderr)
        return 1


def _run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = OUT_DIR / ".tmp-upstream.ttf"

    subprocess.run(["curl", "-sS", "-L", "-o", str(src), UPSTREAM], check=True)
    if src.stat().st_size < 100_000:
        print("upstream download looks wrong; got a redirect page?", file=sys.stderr)
        return 1

    for name, weight in WEIGHTS.items():
        out = build(src, name, weight)
        print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1024:.0f} KB")

    src.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
