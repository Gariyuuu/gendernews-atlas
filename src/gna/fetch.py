"""Streaming ingestion of the AmericanStories corpus (Chronicling America / LOC).

Design notes
------------
The upstream release ships one gzipped tar per year; members are one JSON per
newspaper *scan* (page), stored in date-shuffled order.  We stream the tar over
HTTP and stop once a per-year article quota is met, which yields an
approximately random sample of scans within the year at a small fraction of the
download cost (the 1900-1922 shards are 5-7 GB each).

The shuffled-prefix assumption is not taken on faith:
``scripts/audit_prefix_sampling.py`` compares a streamed prefix against a
complete single-year download.  See research/corpus_provenance.md.

Nothing large is written to disk: tar members are decompressed in flight and
only retained article records are persisted.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import tarfile
import time
from dataclasses import asdict, dataclass
from typing import Iterator

import requests

from .paths import MANIFESTS, RAW, load_config

USER_AGENT = "gendernews-atlas/0.1 (academic research)"
HF_URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/faro_{year}.tar.gz"
PROCESSING_VERSION = "ingest-v2"

_PAGE_RE = re.compile(r"_p(\d+)_")
# Upstream title strings are inconsistent across year shards, e.g.
#   "Evening star."  vs  "Evening star. [volume] (Washington, D.C.) 1854-1972".
# Normalising at ingest keeps `publication` a stable grouping key across decades.
_TITLE_NOISE = re.compile(r"\s*(\[[^\]]*\]|\([^)]*\)|\d{4}\s*-\s*(\d{4}|current|\?{2,4}))\s*", re.I)


def normalize_title(raw: str) -> str:
    t = _TITLE_NOISE.sub(" ", raw or "")
    return " ".join(t.split()).strip(" .,;:-").lower()


def page_from_member(name: str) -> int | None:
    """Page number is reliable in the tar member name; the JSON field is sometimes 'na'."""
    m = _PAGE_RE.search(name or "")
    return int(m.group(1)) if m else None


@dataclass(slots=True)
class Article:
    article_id: str
    publication: str
    publication_raw: str
    lccn: str
    state: str
    date: str
    year: int
    page_number: int | None
    edition: str
    headline: str
    byline: str
    text: str
    n_chars: int
    n_words: int
    legibility: str
    scan_url: str
    scan_id: str
    license: str
    processing_version: str


class _BudgetExhausted(Exception):
    pass


class _CountingReader(io.RawIOBase):
    """Wraps a byte stream and stops it once a download budget is consumed."""

    def __init__(self, raw, max_bytes: int):
        self._raw = raw
        self._max = max_bytes
        self.n = 0

    def readable(self) -> bool:
        return True

    def read(self, size=-1):
        if self.n >= self._max:
            raise _BudgetExhausted
        chunk = self._raw.read(size)
        self.n += len(chunk)
        return chunk


def _iter_tar_members(url: str, max_bytes: int, timeout: int = 180) -> Iterator[tuple[str, bytes]]:
    """Yield (member_name, raw JSON bytes), stopping once `max_bytes` are downloaded."""
    with requests.get(url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=timeout) as resp:
        resp.raise_for_status()
        counter = _CountingReader(resp.raw, max_bytes)
        try:
            with tarfile.open(fileobj=counter, mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".json"):
                        continue
                    fh = tar.extractfile(member)
                    if fh is None:
                        continue
                    yield member.name, fh.read()
        except (_BudgetExhausted, tarfile.ReadError, EOFError, OSError):
            return


def _mk_article_id(scan_id: str, full_article_id) -> str:
    h = hashlib.sha1(f"{scan_id}::{full_article_id}".encode()).hexdigest()[:16]
    return f"as1-{h}"


def parse_scan(member_name: str, payload: bytes, filters: dict, license_str: str) -> list[Article]:
    """Convert one scan JSON into retained Article records."""
    try:
        scan = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    lccn = scan.get("lccn") or {}
    edition = scan.get("edition") or {}
    date = edition.get("date") or ""
    if len(date) < 10 or not date[:4].isdigit():
        return []
    year = int(date[:4])

    page_no = page_from_member(member_name)
    if page_no is None:
        raw_pn = scan.get("page_number")
        page_no = raw_pn if isinstance(raw_pn, int) else None

    title_raw = (lccn.get("title") or "").strip()
    legibility_by_id = {b["id"]: b.get("legibility", "NA") for b in scan.get("bboxes") or []}
    scan_id = (scan.get("scan_url") or "").strip() or f"{lccn.get('lccn', '?')}:{date}:p{page_no}"

    out: list[Article] = []
    for fa in scan.get("full articles") or []:
        text = (fa.get("article") or "").strip()
        if not text:
            continue
        n_chars = len(text)
        if n_chars < filters["min_chars"] or n_chars > filters["max_chars"]:
            continue
        legs = [legibility_by_id.get(i, "NA") for i in (fa.get("object_ids") or [])]
        if "Illegible" in legs:
            leg = "Illegible"
        elif "Questionable" in legs:
            leg = "Questionable"
        elif "Legible" in legs:
            leg = "Legible"
        else:
            leg = "NA"
        if leg not in filters["keep_legibility"]:
            continue
        out.append(
            Article(
                article_id=_mk_article_id(scan_id, fa.get("full_article_id")),
                publication=normalize_title(title_raw),
                publication_raw=title_raw,
                lccn=lccn.get("lccn") or "",
                state=lccn.get("state") or "",
                date=date,
                year=year,
                page_number=page_no,
                edition=edition.get("edition") or "",
                headline=" ".join((fa.get("headline") or "").split()),
                byline=" ".join((fa.get("byline") or "").split()),
                text=text,
                n_chars=n_chars,
                n_words=len(text.split()),
                legibility=leg,
                scan_url=(scan.get("scan_url") or "").strip(),
                scan_id=scan_id,
                license=license_str,
                processing_version=PROCESSING_VERSION,
            )
        )
    return out


def fetch_year(year: int, quota: int, max_mb: int, cfg: dict, force: bool = False) -> dict:
    """Download-and-sample one year; returns a manifest entry."""
    out_path = RAW / f"articles_{year}.jsonl.gz"
    man_path = MANIFESTS / f"year_{year}.json"
    if out_path.exists() and man_path.exists() and not force:
        with open(man_path) as fh:
            man = json.load(fh)
        if man.get("quota") == quota and man.get("processing_version") == PROCESSING_VERSION:
            man["cached"] = True
            return man

    ds = cfg["dataset"]
    url = HF_URL.format(repo=ds["hf_repo"], rev=ds["hf_revision"], year=year)
    filters = cfg["ingest_filters"]
    t0 = time.time()
    n_scans = n_articles = 0
    sha = hashlib.sha256()
    tmp = out_path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for member_name, payload in _iter_tar_members(url, max_bytes=max_mb * 1024 * 1024):
            n_scans += 1
            for art in parse_scan(member_name, payload, filters, ds["license"]):
                line = json.dumps(asdict(art), ensure_ascii=False)
                out.write(line + "\n")
                sha.update(line.encode())
                n_articles += 1
            if n_articles >= quota:
                break
    tmp.replace(out_path)

    man = {
        "year": year,
        "source_url": url,
        "dataset": ds["name"],
        "license": ds["license"],
        "quota": quota,
        "max_mb": max_mb,
        "n_scans_read": n_scans,
        "n_articles": n_articles,
        "quota_met": n_articles >= quota,
        "content_sha256": sha.hexdigest(),
        "processing_version": PROCESSING_VERSION,
        "seconds": round(time.time() - t0, 1),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cached": False,
    }
    with open(man_path, "w") as fh:
        json.dump(man, fh, indent=2)
    return man


def target_years(cfg: dict, tier_name: str) -> list[int]:
    t = cfg["tiers"][tier_name]
    return cfg["year_grid"] if t["years"] == "grid" else t["years"]


def iter_articles(years: list[int] | None = None) -> Iterator[dict]:
    """Stream retained articles from disk, in year order."""
    cfg = load_config("corpus")
    years = years or cfg["year_grid"]
    for y in years:
        p = RAW / f"articles_{y}.jsonl.gz"
        if not p.exists():
            continue
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)
