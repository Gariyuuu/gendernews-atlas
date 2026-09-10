#!/usr/bin/env python
"""Append reference labels given in compact form (one item per line) to a labels JSONL.

Line format:  n|is_person|gender|ROLES|quoted|confidence|note
  is_person: y/n/u   gender: F/M/U   ROLES: comma codes (PO MI BU LA PR AS CI SO FA CR) or empty
  quoted: y/n        confidence: 1-3
Items are keyed by the blind file's sequence number n, mapped to entity_id here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "annotation"
CODES = {"PO": "PUBLIC_OFFICE", "MI": "MILITARY", "BU": "BUSINESS", "LA": "LABOR", "PR": "PROFESSIONAL",
         "AS": "ARTS_SPORTS", "CI": "CIVIC", "SO": "SOCIAL", "FA": "FAMILY", "CR": "CRIME_ACCIDENT"}


def main() -> int:
    out = ANN / (sys.argv[1] if len(sys.argv) > 1 else "labels_v1.jsonl")
    pass_no = 2 if "consistency" in out.name else 1
    by_n = {json.loads(x)["n"]: json.loads(x)["entity_id"] for x in open(ANN / "sample_blind.jsonl")}
    have = {json.loads(x)["entity_id"] for x in open(out)} if out.exists() else set()
    added = 0
    with open(out, "a") as fh:
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            n, p, g, roles, q, c, *note = line.split("|")
            eid = by_n[int(n)]
            if eid in have:
                raise SystemExit(f"duplicate label for n={n}")
            rec = {"entity_id": eid, "n": int(n), "is_person": {"y": "yes", "n": "no", "u": "unsure"}[p],
                   "gender_text": {"F": "F", "M": "M", "U": "UNKNOWN"}[g],
                   "roles": [CODES[r] for r in roles.split(",") if r],
                   "quoted": {"y": "yes", "n": "no"}[q], "confidence": int(c), "notes": "|".join(note).strip(),
                   "annotator": "AI build agent (Claude Opus 5); not human", "protocol": "v1", "pass": pass_no}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            have.add(eid)
            added += 1
    print(f"added {added} -> {out.name} (total {len(have)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
