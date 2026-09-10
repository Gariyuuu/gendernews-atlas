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


def free_gb(path: Path = ROOT) -> float:
    import shutil
    return shutil.disk_usage(path).free / 1e9


def wait_for_space(min_gb: float | None = None, max_wait_s: int = 3600, poll_s: int = 30) -> None:
    """Block until the volume has `min_gb` free.  The build machine's disk is shared with
    other jobs and routinely hovers near 0 GB free; a transient full disk should pause,
    not corrupt, a long run."""
    import time
    if min_gb is None:
        min_gb = float(os.environ.get("GNA_MIN_FREE_GB", "0.5"))
    waited = 0
    while free_gb() < min_gb and waited < max_wait_s:
        print(f"[disk] {free_gb():.2f} GB free < {min_gb} GB; waiting {poll_s}s", flush=True)
        time.sleep(poll_s)
        waited += poll_s


def safe_write(write_fn, path, retries: int = 20, poll_s: int = 30) -> None:
    """Call write_fn(tmp_path), then atomically rename; retry on ENOSPC."""
    import errno
    import time
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    for attempt in range(retries):
        wait_for_space()
        try:
            write_fn(tmp)
            tmp.replace(path)
            return
        except OSError as e:
            tmp.unlink(missing_ok=True)
            if e.errno != errno.ENOSPC or attempt == retries - 1:
                raise
            print(f"[disk] ENOSPC writing {path.name}; retry {attempt + 1}/{retries}", flush=True)
            time.sleep(poll_s)
