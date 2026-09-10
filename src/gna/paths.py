from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
MANIFESTS = DATA / "manifests"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESEARCH = ROOT / "research"
PAPER = ROOT / "paper"
SITE_DATA = ROOT / "site" / "public" / "results"

for _p in (RAW, INTERIM, MANIFESTS, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)


def load_config(name: str) -> dict:
    with open(CONFIG / f"{name}.json") as fh:
        return json.load(fh)


def tier() -> str:
    return os.environ.get("GNA_TIER", "standard").lower()
