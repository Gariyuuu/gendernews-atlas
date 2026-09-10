#!/usr/bin/env python
"""Test whether a streamed tar prefix is a representative sample of a year.

Two checks:
  1. FULL-YEAR (1963): stream the entire shard; compare the prefix the sampler used
     against the full-year distribution of newspaper (LCCN), month, and page.
  2. DEEP-STREAM (1900, 5.7 GB shard): stream a large budget; compare the prefix
     against the later part of the stream (ordering homogeneity).

In both, the observed total-variation distance (TVD) is compared with the TVD of
random samples of the same size (null distribution for representative sampling).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gna.fetch import HF_URL, _iter_tar_members, page_from_member  # noqa: E402
from gna.paths import MANIFESTS, RESULTS, load_config  # noqa: E402


def scan_keys(year: int, max_mb: int):
    cfg = load_config("corpus")["dataset"]
    url = HF_URL.format(repo=cfg["hf_repo"], rev=cfg["hf_revision"], year=year)
    rows = []
    for name, _ in _iter_tar_members(url, max_bytes=max_mb * 1024 * 1024):
        base = name.rsplit("/", 1)[-1]
        parts = base.split("_")
        # 1963-12-15_p23_sn83045462_00280609249_1963121501_0333.json
        rows.append({"lccn": parts[2] if len(parts) > 2 else "?", "month": base[5:7],
                     "page_band": _band(page_from_member(base))})
    return rows


def _band(p):
    if p is None:
        return "na"
    return "p1" if p == 1 else ("p2-4" if p <= 4 else ("p5-12" if p <= 12 else "p13+"))


def tvd(a: list, b: list) -> float:
    ca, cb = collections.Counter(a), collections.Counter(b)
    na, nb = len(a), len(b)
    keys = set(ca) | set(cb)
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in keys)


def compare(sample: list[dict], population: list[dict], rng, n_null: int = 500) -> dict:
    out = {}
    n = len(sample)
    for field in ("lccn", "month", "page_band"):
        obs = tvd([r[field] for r in sample], [r[field] for r in population])
        pop = [r[field] for r in population]
        null = np.array([tvd(list(rng.choice(pop, size=n, replace=False)), pop) for _ in range(n_null)])
        out[field] = {
            "tvd_observed": round(obs, 4),
            "tvd_null_median": round(float(np.median(null)), 4),
            "tvd_null_p95": round(float(np.quantile(null, 0.95)), 4),
            "p_value_one_sided": round(float((null >= obs).mean()), 4),
            "within_null_95": bool(obs <= np.quantile(null, 0.95)),
        }
    return out


def main() -> int:
    rng = np.random.default_rng(20260910)
    report = {}

    y = 1963
    used = json.load(open(MANIFESTS / f"year_{y}.json"))["n_scans_read"]
    full = scan_keys(y, max_mb=10_000)
    report["full_year_1963"] = {
        "n_scans_full_year": len(full),
        "n_scans_prefix_used": used,
        "comparison_prefix_vs_full": compare(full[:used], full, rng),
        "top_lccn_share_full": collections.Counter(r["lccn"] for r in full).most_common(1)[0][1] / len(full),
        "top_lccn_share_prefix": collections.Counter(r["lccn"] for r in full[:used]).most_common(1)[0][1] / used,
    }
    print(json.dumps(report["full_year_1963"], indent=1), flush=True)

    y = 1900
    used = json.load(open(MANIFESTS / f"year_{y}.json"))["n_scans_read"]
    deep = scan_keys(y, max_mb=700)
    later = deep[used:]
    report["deep_stream_1900"] = {
        "n_scans_streamed": len(deep),
        "n_scans_prefix_used": used,
        "note": "population = all scans in the first 700 MB of a 5.7 GB shard; tests ordering homogeneity, not full-year representativeness",
        "comparison_prefix_vs_streamed": compare(deep[:used], deep, rng),
        "comparison_prefix_vs_later_only": {k: round(tvd([r[k] for r in deep[:used]], [r[k] for r in later]), 4)
                                            for k in ("lccn", "month", "page_band")},
    }
    print(json.dumps(report["deep_stream_1900"], indent=1), flush=True)
    with open(RESULTS / "sampling_audit.json", "w") as fh:
        json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
