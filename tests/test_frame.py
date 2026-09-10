import numpy as np
import pandas as pd

from gna.frame import cap_flag, is_f, page_band, role_flags


def test_role_flags_and_authority_composite():
    r = pd.Series([["PUBLIC_OFFICE"], ["CIVIC", "FAMILY"], [], ["BUSINESS", "PROFESSIONAL"]])
    f = role_flags(r, "B")
    assert f["B_PUBLIC_OFFICE"].tolist() == [True, False, False, False]
    assert f["B_AUTHORITY"].tolist() == [True, False, False, True]      # CIVIC is not AUTHORITY
    assert f["B_FAMILY"].tolist() == [False, True, False, False]


def test_is_f_keeps_unknown_missing():
    out = is_f(pd.Series(["F", "M", "UNKNOWN"]))
    assert out.iloc[0] == 1.0 and out.iloc[1] == 0.0 and np.isnan(out.iloc[2])


def test_page_band():
    assert [page_band(x) for x in (1, 3, 9, 20, None)] == ["p1", "p2-4", "p5-12", "p13+", "na"]


def test_cap_flag_is_deterministic_and_caps_per_paper_year():
    ids = [f"as1-{i:016x}" for i in range(50)]
    a = pd.DataFrame({"article_id": ids, "publication": ["x"] * 30 + ["y"] * 20, "year": [1900] * 50})
    f1, f2 = cap_flag(a, cap=10), cap_flag(a.sample(frac=1, random_state=3).sort_index(), cap=10)
    assert f1.sum() == 20 and (f1 == f2).all()
