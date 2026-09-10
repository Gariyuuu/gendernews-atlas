#!/usr/bin/env python
"""make data -- stream and sample the AmericanStories corpus by year."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gna.fetch import fetch_year, target_years  # noqa: E402
from gna.paths import MANIFESTS, load_config, tier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=tier())
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = load_config("corpus")
    t = cfg["tiers"][a.tier]
    years = a.years or target_years(cfg, a.tier)
    rows = []
    for y in years:
        man = fetch_year(y, t["articles_per_year"], t["max_mb_per_year"], cfg, force=a.force)
        rows.append(man)
        flag = "cached" if man.get("cached") else f"{man['seconds']}s"
        print(f"{y}  scans={man['n_scans_read']:>6}  articles={man['n_articles']:>7}  {flag}", flush=True)
    total = sum(r["n_articles"] for r in rows)
    with open(MANIFESTS / "corpus_manifest.json", "w") as fh:
        json.dump({"tier": a.tier, "years": years, "total_articles": total, "entries": rows}, fh, indent=2)
    print(f"TOTAL {total} articles across {len(years)} years")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
