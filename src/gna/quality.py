"""Corpus quality: exact and near-duplicate detection, OCR-quality proxy, length drift."""
from __future__ import annotations

import gzip
import hashlib
import re
from collections import defaultdict

import numpy as np
from datasketch import MinHash, MinHashLSH

from .paths import CONFIG

_TOK = re.compile(r"[a-z0-9]+")


def load_lexicon() -> set[str]:
    with gzip.open(CONFIG / "lexicons" / "web2.txt.gz", "rt") as fh:
        return {w.strip().lower() for w in fh if w.strip()}


def exact_key(text: str) -> str:
    return hashlib.sha1(" ".join(_TOK.findall(text.lower())).encode()).hexdigest()


def shingles(text: str, k: int = 5) -> set[bytes]:
    toks = _TOK.findall(text.lower())
    if len(toks) < k:
        return {" ".join(toks).encode()} if toks else set()
    return {" ".join(toks[i:i + k]).encode() for i in range(len(toks) - k + 1)}


class _UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def dedupe(records: list[tuple[str, str]], threshold: float = 0.8, num_perm: int = 128,
           seed: int = 1) -> dict[str, dict]:
    """records: (article_id, text).  Returns id -> {exact_group, near_group}.

    Exact groups share a normalised-token hash; near groups are connected components
    of the MinHash-LSH graph (estimated Jaccard >= threshold on 5-word shingles).
    Group ids are the lexicographically smallest member id, so results are order-free.
    """
    uf_exact, uf_near = _UF(), _UF()
    by_key: dict[str, str] = {}
    for aid, text in records:
        k = exact_key(text)
        if k in by_key:
            uf_exact.union(aid, by_key[k])
            uf_near.union(aid, by_key[k])
        else:
            by_key[k] = aid
        uf_exact.find(aid)
        uf_near.find(aid)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    reps = set(by_key.values())
    for aid, text in records:
        if aid not in reps:
            continue
        sh = shingles(text)
        if not sh:
            continue
        mh = MinHash(num_perm=num_perm, seed=seed)
        mh.update_batch(list(sh))
        for other in lsh.query(mh):
            uf_near.union(aid, other)
        lsh.insert(aid, mh)
    return {aid: {"exact_group": uf_exact.find(aid), "near_group": uf_near.find(aid)} for aid, _ in records}


def group_flags(groups: dict[str, dict], order: dict[str, tuple]) -> dict[str, dict]:
    """Mark every member except the earliest (by `order` key) as a duplicate."""
    out = {}
    for kind in ("exact_group", "near_group"):
        members = defaultdict(list)
        for aid, g in groups.items():
            members[g[kind]].append(aid)
        for gid, ids in members.items():
            ids.sort(key=lambda a: order[a])
            for i, aid in enumerate(ids):
                out.setdefault(aid, {})[kind.replace("_group", "_dup")] = i > 0
                out[aid][kind.replace("_group", "_size")] = len(ids)
                out[aid][kind] = gid
    return out


def junk_rate(text: str) -> float:
    if not text:
        return float("nan")
    bad = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,;:'\"-()?!$%&"))
    return bad / len(text)


def hhi(counts) -> float:
    c = np.asarray(list(counts), dtype=float)
    s = c.sum()
    return float(((c / s) ** 2).sum()) if s else float("nan")
