import numpy as np
import pandas as pd
import pytest

from gna.models import bh_fdr, hac_trend, log_odds_dirichlet, lpm_trend, unknown_bounds, yearly_share


def test_yearly_share_point_estimate_and_ci_cover():
    rng = np.random.default_rng(0)
    rows = []
    for y, p in [(1900, 0.2), (1930, 0.4)]:
        for a in range(400):
            for _ in range(3):
                rows.append({"year": y, "article_id": f"{y}-{a}", "s": rng.random() < p})
    df = pd.DataFrame(rows)
    out = yearly_share(df, "s", B=300, seed=1).set_index("year")
    for y, p in [(1900, 0.2), (1930, 0.4)]:
        assert abs(out.loc[y, "share"] - p) < 0.04
        assert out.loc[y, "ci_lo"] < out.loc[y, "share"] < out.loc[y, "ci_hi"]
        assert out.loc[y, "n_clusters"] == 400 and out.loc[y, "n"] == 1200


def test_hac_trend_recovers_linear_slope_in_pp_per_decade():
    years = np.arange(1900, 1964, 3)
    share = 0.10 + 0.02 * (years - 1930) / 10
    t = hac_trend(pd.DataFrame({"year": years, "share": share, "n": 1000.0}))
    assert t["slope_pp_dec"] == pytest.approx(2.0, abs=1e-6)
    assert t["level_1930"] == pytest.approx(10.0, abs=1e-6)


def test_fixed_effects_remove_composition_confound():
    # Paper A: early years, y ~ 0.6; paper B: late years, y ~ 0.2.  Within each paper: flat.
    rng = np.random.default_rng(2)
    rows = []
    for y in range(1900, 1964, 3):
        share_b = (y - 1900) / 63
        for i in range(600):
            pub = "B" if rng.random() < share_b else "A"
            p = 0.2 if pub == "B" else 0.6
            rows.append({"year": y, "publication": pub, "y": float(rng.random() < p)})
    df = pd.DataFrame(rows)
    raw = lpm_trend(df, "y", fe=None, cluster="publication", cluster2="year")
    adj = lpm_trend(df, "y", fe=["publication"], cluster="publication", cluster2="year")
    assert raw["slope_pp_dec"] < -4                    # composition makes it look like a decline
    assert abs(adj["slope_pp_dec"]) < 1.0              # within-paper: no trend


def test_bh_fdr_matches_hand_computation():
    q = bh_fdr(np.array([0.01, 0.04, 0.03, 0.2, np.nan]))
    assert np.allclose(q[:4], [0.04, 0.04 / 0.75, 0.04 / 0.75, 0.2])
    assert np.isnan(q[4])


def test_unknown_bounds():
    b = unknown_bounds(10, 30, 60)
    assert b["share_f_classified"] == 0.25 and b["lower"] == 0.1 and b["upper"] == 0.7


def test_log_odds_sign():
    a = pd.Series({"lead": 50, "say": 100, "suffer": 2})
    b = pd.Series({"lead": 5, "say": 100, "suffer": 30})
    out = log_odds_dirichlet(a, b).set_index(pd.Index(["suffer", "say", "lead"])) if False else log_odds_dirichlet(a, b)
    assert out.loc["lead", "z"] > 2 and out.loc["suffer", "z"] < -2
