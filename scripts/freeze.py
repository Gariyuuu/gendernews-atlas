#!/usr/bin/env python
"""make freeze -- write results/release.json: provenance hashes + headline numbers.

Everything the site and paper quote comes from here or from the parquet files it
hashes.  Re-running on an unchanged tree reproduces the same hashes.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gna.paths import CONFIG, DATA, FIGURES, MANIFESTS, PAPER, RESULTS, ROOT, safe_write  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(d: Path, pattern: str) -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(d.glob(pattern)) if p.is_file()}


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _f(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return None if np.isnan(x) else round(x, 4)


def headline() -> dict:
    """Numbers quoted in prose.  Each entry names its source file."""
    h: dict = {}
    cs = pd.read_parquet(RESULTS / "corpus_summary.parquet")
    man = json.load(open(MANIFESTS / "corpus_manifest.json"))
    h["corpus"] = {"source": "results/corpus_summary.parquet + data/manifests/corpus_manifest.json",
                   "years": [int(y) for y in cs["year"]], "n_years": int(len(cs)),
                   "n_articles": int(cs["n_articles"].sum()), "first_year": int(cs["year"].min()),
                   "last_year": int(cs["year"].max()), "tier": man["tier"],
                   "n_publications_pre1922_mean": _f(cs.loc[cs["year"] <= 1921, "n_publications"].mean()),
                   "n_publications_post1924_mean": _f(cs.loc[cs["year"] >= 1924, "n_publications"].mean()),
                   "evening_star_share_first": _f(cs["evening_star_share"].iloc[0]),
                   "evening_star_share_last": _f(cs["evening_star_share"].iloc[-1]),
                   "near_dup_share_overall": _f((cs["near_dup_share"] * cs["n_articles"]).sum() / cs["n_articles"].sum())}
    if (RESULTS / "sampling_audit.json").exists():
        sa = json.load(open(RESULTS / "sampling_audit.json"))["full_year_1963"]["comparison_prefix_vs_full"]
        h["sampling_audit"] = {"source": "results/sampling_audit.json",
                               **{f"{k}_p": v["p_value_one_sided"] for k, v in sa.items()}}
    ex = RESULTS / "extraction_summary.parquet"
    if ex.exists():
        e = pd.read_parquet(ex)
        h["extraction"] = {"source": str(ex.relative_to(ROOT)), "n_entities_ner": int(e["n_entities_ner"].sum()),
                           "unknown_share_hp_first": _f(e["unknown_share_hp"].iloc[0]),
                           "unknown_share_hp_last": _f(e["unknown_share_hp"].iloc[-1]),
                           "unknown_share_hp_min": _f(e["unknown_share_hp"].min()),
                           "unknown_share_hp_max": _f(e["unknown_share_hp"].max())}
    ts = RESULTS / "trend_summary.parquet"
    if ts.exists():
        t = pd.read_parquet(ts)
        pick = t[(t["estimand"].isin(["female_share", "female_share_in_role"])) & (t["population"] == "ner")]
        h["trends"] = {"source": str(ts.relative_to(ROOT)), "unit": "percentage points per decade (HAC 95% CI)",
                       "rows": [{k: _f(v) if not isinstance(v, str) else v for k, v in r.items()
                                 if k in ("estimand", "role", "method", "tier", "slope_pp_dec", "ci_lo", "ci_hi", "p",
                                          "early_share", "late_share", "early_late_diff_pp")}
                                for r in pick.to_dict("records")]}
    gc = RESULTS / "gender_consistency.parquet"
    if gc.exists():
        g = pd.read_parquet(gc)
        tot = g[g["year"] == -1].iloc[0]
        h["gender_consistency"] = {"source": str(gc.relative_to(ROOT)), "n": int(tot["n"]), "agree": _f(tot["agree"]),
                                   "agree_when_F": _f(tot["agree_when_F"]), "agree_when_M": _f(tot["agree_when_M"])}
    for name in ("validation", "topic_adjusted", "method_agreement", "robustness", "quote_trends", "language"):
        p = RESULTS / f"{name}.parquet"
        if p.exists():
            h.setdefault("available", []).append(str(p.relative_to(ROOT)))
    return h


def main() -> int:
    llm_meta = {}
    for p in (DATA / "annotation" / "llm_reference.jsonl", DATA / "interim" / "llm_apply.jsonl"):
        if p.exists():
            recs = [json.loads(x) for x in open(p) if x.strip()]
            llm_meta[str(p.relative_to(ROOT))] = {
                "n": len(recs), "n_parse_ok": sum(bool(r.get("parse_ok")) for r in recs),
                "models": sorted({r.get("model", "?") for r in recs}),
                "prompt_hashes": sorted({r.get("prompt_hash", "?") for r in recs}),
                "first_ts": min((r.get("timestamp", "") for r in recs), default=""),
                "last_ts": max((r.get("timestamp", "") for r in recs), default="")}
    rel = {
        "project": "GenderNews Atlas",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
        "corpus": {"manifest_sha256": sha(MANIFESTS / "corpus_manifest.json"),
                   "per_year_content_sha256": {p.stem: json.load(open(p))["content_sha256"]
                                               for p in sorted(MANIFESTS.glob("year_*.json"))}},
        "software": {"python": platform.python_version(), "spacy": ver("spacy"), "en_core_web_sm": ver("en_core_web_sm"),
                     "scikit-learn": ver("scikit-learn"), "statsmodels": ver("statsmodels"), "torch": ver("torch"),
                     "transformers": ver("transformers"), "pandas": ver("pandas"), "datasketch": ver("datasketch")},
        "models": {"ner_parser": f"en_core_web_sm {ver('en_core_web_sm')}",
                   "encoder_D": "sentence-transformers/all-MiniLM-L6-v2", "llm_E": llm_meta},
        "config_sha256": {**hash_dir(CONFIG, "*.json"), **hash_dir(ROOT / "src" / "gna", "*.py"),
                          "research/hypotheses.md": sha(ROOT / "research" / "hypotheses.md"),
                          "research/measurement_framework.md": sha(ROOT / "research" / "measurement_framework.md")},
        "annotation_sha256": hash_dir(DATA / "annotation", "*"),
        "results_sha256": {k: v for k, v in hash_dir(RESULTS, "*").items() if not k.endswith("release.json")},
        "figures_sha256": hash_dir(FIGURES, "*"),
        "paper_sha256": hash_dir(PAPER, "*.md"),
        "headline": headline(),
    }
    safe_write(lambda t: Path(t).write_text(json.dumps(rel, indent=1, ensure_ascii=False)), RESULTS / "release.json")
    print(f"release.json written: {len(rel['results_sha256'])} results, {len(rel['figures_sha256'])} figures, "
          f"git {rel['git']['sha'][:8]}{' (dirty)' if rel['git']['dirty'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
