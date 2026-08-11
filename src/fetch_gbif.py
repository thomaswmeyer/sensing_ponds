"""Download GBIF occurrence photos for floating aquatic plant species.

Builds a geographically diverse training set to complement WaterHyacinth, which
is confined to two Bangladeshi districts over three months. Deployment target is
Tamil-speaking (Tamil Nadu / Sri Lanka), so regional records are prioritised.

Writes a folder-per-class tree matching what train_mobile_classifier.py expects,
plus a manifest.csv carrying provenance -- licence, attribution, country and
GBIF occurrence key -- for every image. CC BY requires attribution, so the
manifest is a licence compliance record, not a convenience.

Usage:
    python src/fetch_gbif.py --out data/gbif --per-species 800
    python src/fetch_gbif.py --out data/gbif --regional-only    # IN + LK only
    python src/fetch_gbif.py --out data/gbif --dry-run          # counts only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

API = "https://api.gbif.org/v1/occurrence/search"
PAGE = 300  # GBIF max limit per request

# Licences safe for a distributed commercial product. NC and ND are excluded:
# CC-BY-NC blocks commercial use, ND blocks derivative works (a trained model
# is arguably derivative). Widening this is a legal decision, not a technical one.
SAFE_LICENCES = ["CC0_1_0", "CC_BY_4_0"]

# Deployment region. GBIF ISO country codes.
REGION = ["IN", "LK"]


@dataclass(frozen=True)
class Species:
    """A target class.

    taxon_key resolved via /species/match on 2026-08-11. Verify with --dry-run
    if these ever look wrong -- GBIF backbone keys do change across releases.
    """

    folder: str
    taxon_key: int
    label: str
    note: str = ""


SPECIES = [
    # Accepted name is Pontederia crassipes; Eichhornia crassipes (key 2765940)
    # is a SYNONYM. Querying the accepted key returns synonym records too, so
    # one key is sufficient -- do not query both or you double-count.
    Species("water_hyacinth", 2765942, "Pontederia crassipes",
            "syn. Eichhornia crassipes; ~22.5k imaged occurrences"),
    Species("water_lettuce", 2870583, "Pistia stratiotes", "~12.8k"),
    Species("duckweed", 2867589, "Lemna minor", "~17.8k"),
    # Replaces the Mendeley dataset's 4th class (Monochoria korsakowii), which is
    # unusable here: temperate East Asian, 0 records in IN/LK, and only 111 imaged
    # occurrences globally. Its congener M. vaginalis is common in South Asia but
    # yields just 6 CC0/CC-BY field photos -- nearly all GBIF records for it are
    # PRESERVED_SPECIMEN (herbarium sheets), useless as field-photo training data.
    #
    # Salvinia is the better 4th class: a documented hyacinth confuser in the
    # remote sensing literature, present in the deployment region, and adequately
    # imaged. Trains from GBIF alone -- no Mendeley coverage.
    Species("salvinia", 5274863, "Salvinia molesta",
            "replaces Monochoria; GBIF-only, no Mendeley coverage"),
]

IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp)(\?|$)", re.I)


CURL = shutil.which("curl")
UA = "sensing-pondy/0.1"


def _curl(url: str, timeout: int, binary: bool) -> bytes:
    """Fetch via curl rather than urllib.

    Deliberate: this machine sits behind a TLS-inspecting proxy whose root CA is
    in the macOS keychain but not in Python's certifi bundle, so urllib fails
    with CERTIFICATE_VERIFY_FAILED on every HTTPS call. curl uses the system
    trust store and works. Shelling out avoids touching the cert configuration.

    If this ever runs on a machine without such a proxy, urllib would be fine --
    but curl is portable enough that there is no reason to branch.
    """
    if not CURL:
        raise RuntimeError("curl not found on PATH; required for HTTPS on this machine")
    proc = subprocess.run(
        [CURL, "-sS", "--fail", "--location", "--max-time", str(timeout),
         "-A", UA, url],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"curl {proc.returncode}: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout if binary else proc.stdout


def api_get(params: dict, retries: int = 4) -> dict:
    """GET with backoff. GBIF rate-limits aggressively on burst traffic."""
    url = f"{API}?{urllib.parse.urlencode(params, doseq=True)}"
    for attempt in range(retries):
        try:
            return json.loads(_curl(url, timeout=60, binary=False))
        except (RuntimeError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"GBIF request failed after {retries} tries: {exc}") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def base_query(sp: Species, regional: bool, safe_licences: bool) -> dict:
    q = {
        "taxonKey": sp.taxon_key,
        "mediaType": "StillImage",
        # Research-grade proxy: >=2 identifiers with >2/3 agreement. Still crowd
        # consensus, not expert verification -- expect residual label noise.
        "basisOfRecord": "HUMAN_OBSERVATION",
        "hasCoordinate": "true",
    }
    if regional:
        q["country"] = REGION
    if safe_licences:
        q["license"] = SAFE_LICENCES
    return q


def count_for(sp: Species, regional: bool, safe_licences: bool) -> int:
    return api_get({**base_query(sp, regional, safe_licences), "limit": 0})["count"]


def harvest_records(sp: Species, target: int, regional: bool, safe_licences: bool) -> list[dict]:
    """Page through occurrences, preferring regional records.

    Regional images are pulled first and topped up globally. A model trained
    only on global records will underperform on South Indian water bodies; one
    trained only on the ~150 regional records will overfit. Prefer-then-top-up
    gets both.
    """
    out: list[dict] = []
    seen_keys: set[int] = set()

    passes = [True, False] if regional else [False]
    for regional_pass in passes:
        if len(out) >= target:
            break
        offset = 0
        while len(out) < target:
            data = api_get({
                **base_query(sp, regional_pass, safe_licences),
                "limit": PAGE,
                "offset": offset,
            })
            results = data.get("results", [])
            if not results:
                break

            for rec in results:
                key = rec.get("key")
                if key in seen_keys:
                    continue
                media = [
                    m for m in rec.get("media", [])
                    if m.get("identifier") and IMAGE_EXT_RE.search(m["identifier"])
                ]
                if not media:
                    continue
                seen_keys.add(key)
                # One image per occurrence: extra photos of the same plant on the
                # same day are near-duplicates and would leak across the split.
                out.append({
                    "gbif_key": key,
                    "url": media[0]["identifier"],
                    "license": rec.get("license", ""),
                    "rights_holder": rec.get("rightsHolder", ""),
                    "country": rec.get("countryCode", ""),
                    "lat": rec.get("decimalLatitude", ""),
                    "lon": rec.get("decimalLongitude", ""),
                    "event_date": rec.get("eventDate", ""),
                    "recorded_by": rec.get("recordedBy", ""),
                    "regional": regional_pass,
                })
                if len(out) >= target:
                    break

            offset += PAGE
            if data.get("endOfRecords") or offset >= 100_000:
                break
            time.sleep(0.2)
    return out


def download_one(rec: dict, dest: Path) -> tuple[bool, str]:
    path = dest / f"gbif_{rec['gbif_key']}.jpg"
    if path.exists():
        return True, "cached"
    try:
        blob = _curl(rec["url"], timeout=45, binary=True)
        # Guard against HTML error pages served with a 200. Check magic bytes
        # rather than trusting the URL extension or Content-Type.
        if not (blob.startswith(b"\xff\xd8\xff") or blob.startswith(b"\x89PNG")
                or blob[:4] == b"RIFF"):
            return False, "not-an-image"
        if len(blob) < 4096:
            return False, "too-small"
        path.write_bytes(blob)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - third-party image hosts fail in many ways
        return False, type(exc).__name__


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    safe = not args.any_licence

    print(f"Licences: {'CC0 / CC-BY only' if safe else 'ALL (review before shipping)'}")
    print(f"Region:   {'IN + LK only' if args.regional_only else 'regional-first, global top-up'}")
    print()

    print(f"{'class':<18}{'species':<26}{'global':>9}{'regional':>10}")
    print("-" * 63)
    plans: list[tuple[Species, int]] = []
    for sp in SPECIES:
        g = count_for(sp, False, safe)
        time.sleep(0.3)
        r = count_for(sp, True, safe)
        time.sleep(0.3)
        available = r if args.regional_only else g
        plans.append((sp, min(args.per_species, available)))
        print(f"{sp.folder:<18}{sp.label:<26}{g:>9,}{r:>10,}")
        if sp.note:
            print(f"{'':<18}\033[2m{sp.note}\033[0m")

    if args.dry_run:
        print("\n--dry-run: nothing downloaded.")
        return

    print()
    manifest_rows: list[dict] = []
    for sp, target in plans:
        dest = out_root / sp.folder
        dest.mkdir(parents=True, exist_ok=True)

        records = harvest_records(
            sp, target, regional=not args.global_only, safe_licences=safe
        )
        if args.regional_only:
            records = [r for r in records if r["regional"]]

        ok = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_one, r, dest): r for r in records}
            for i, fut in enumerate(as_completed(futures), 1):
                rec = futures[fut]
                success, status = fut.result()
                if success:
                    ok += 1
                    manifest_rows.append({**rec, "species": sp.label, "class": sp.folder})
                print(f"\r  {sp.folder:<18} {ok}/{len(records)} downloaded", end="", flush=True)
        n_regional = sum(1 for r in manifest_rows if r["class"] == sp.folder and r["regional"])
        print(f"\r  {sp.folder:<18} {ok}/{len(records)} downloaded ({n_regional} regional)")

    manifest = out_root / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class", "species", "gbif_key", "license", "rights_holder",
                        "recorded_by", "country", "lat", "lon", "event_date",
                        "regional", "url"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest: {manifest}  ({len(manifest_rows)} images)")
    print(
        "\nATTRIBUTION: CC-BY images require crediting rights_holder / recorded_by.\n"
        "Keep manifest.csv with any redistributed model or dataset.\n"
        "\nNEXT: review a sample by eye before training. Citizen-science photos\n"
        "include herbarium sheets, close-ups of flowers, habitat shots with no\n"
        "plant visible, and occasional misidentifications."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="data/gbif")
    p.add_argument("--per-species", type=int, default=800)
    p.add_argument("--regional-only", action="store_true",
                   help="India + Sri Lanka only (small but on-domain)")
    p.add_argument("--global-only", action="store_true",
                   help="Skip regional prioritisation")
    p.add_argument("--any-licence", action="store_true",
                   help="Include NC/ND licences -- NOT safe for a shipped product")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", help="Print counts only")
    return p.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
