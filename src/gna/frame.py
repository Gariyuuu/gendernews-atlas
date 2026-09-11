"""The entity-level analysis frame shared by every analysis script.

One row per person-in-article with article metadata, quality flags, topic and
precomputed role indicators for methods A and B.  Heavy list columns (verbs,
modifiers) are split into a separate language frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lexicons import AUTHORITY, ROLES
from .paths import INTERIM

FRAME = INTERIM / "frame.parquet"
LANG = INTERIM / "frame_language.parquet"
ROLE_COLS = ROLES + ["AUTHORITY"]
TIERS = ("h", "hp", "hpn")
CLEAN_DICT_RATE = 0.75          # pre-registered OCR filter (measurement_framework.md §12)
PAPER_YEAR_CAP = 500            # articles per newspaper-year in the "capped" spec


def role_flags(role_lists: pd.Series, prefix: str) -> pd.DataFrame:
    sets = role_lists.map(lambda r: set(r) if r is not None else set())
    out = {f"{prefix}_{r}": sets.map(lambda s, r=r: r in s).astype(bool) for r in ROLES}
    out[f"{prefix}_AUTHORITY"] = sets.map(lambda s: bool(s & AUTHORITY)).astype(bool)
    return pd.DataFrame(out, index=role_lists.index)


# Honorifics that mark a woman, and name suffixes that are not titles.
F_HONORIFICS = frozenset({"mrs", "miss", "misses", "ms", "madame", "mme", "mlle", "lady"})
NAME_SUFFIXES = frozenset({"jr", "sr"})


def mixed_title(titles) -> bool:
    """True when a cluster joins a female honorific with any other title.

    In the couple construction ("Capt. Clayton Bissell ... Mrs. Bissell") a surname-only
    "Mrs." mention attaches to the husband's titled full name (decision D17).  Such a
    cluster is not one person, so its honorific cannot sign its gender.
    """
    s = set(titles) if titles is not None else set()
    return bool(s & F_HONORIFICS) and bool(s - F_HONORIFICS - NAME_SUFFIXES)


def apply_mixed_title_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of `df` with every gender tier set to UNKNOWN on mixed-title clusters; adds `mixed_title`."""
    df = df.copy()
    df["mixed_title"] = df["titles"].map(mixed_title).astype(bool)
    for c in ("gender_h", "gender_hp", "gender_hpn"):
        if c in df:
            df.loc[df["mixed_title"], c] = "UNKNOWN"
    return df


def is_f(g: pd.Series) -> pd.Series:
    """1.0 for F, 0.0 for M, NaN for UNKNOWN / conflict."""
    return g.map({"F": 1.0, "M": 0.0}).astype(float)


def page_band(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "na"
    p = int(p)
    return "p1" if p == 1 else ("p2-4" if p <= 4 else ("p5-12" if p <= 12 else "p13+"))


def cap_flag(articles: pd.DataFrame, cap: int = PAPER_YEAR_CAP) -> pd.Series:
    """Deterministic per-(paper, year) cap: keep the `cap` articles with the smallest id hash."""
    h = articles["article_id"].map(lambda a: int(a.split("-")[1][:8], 16))
    rank = h.groupby([articles["publication"], articles["year"]]).rank(method="first")
    return rank <= cap


def load(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(FRAME, columns=columns)
