import math

from gna.fetch import normalize_title, page_from_member, parse_scan
from gna.quality import dedupe, exact_key, group_flags, hhi, junk_rate, shingles
from gna.textnorm import dictionary_rate, normalize


def test_normalize_joins_linebreak_hyphenation_and_flattens():
    assert normalize("Miss Bur-\nchard studied\nhome economics") == "Miss Burchard studied home economics"


def test_normalize_repairs_ocr_title_dots_and_quotes():
    assert normalize("Mr.. Frohman said ,,yes'' today") == 'Mr. Frohman said "yes" today'
    assert normalize("“No,” she said") == '"No," she said'


def test_dictionary_rate_handles_inflections_and_empty():
    lex = {"walk", "the", "dog", "bake"}
    rate, n = dictionary_rate("The dog walked; baking xqzt", lex)
    assert n == 5 and math.isclose(rate, 4 / 5)
    r0, n0 = dictionary_rate("", lex)
    assert n0 == 0 and math.isnan(r0)


def test_title_normalisation_is_stable_across_shards():
    a = normalize_title("Evening star. [volume] (Washington, D.C.) 1854-1972")
    b = normalize_title("Evening star.")
    assert a == b == "evening star"
    assert normalize_title("The central record. 18??-current") == "the central record"


def test_page_number_from_member_name():
    assert page_from_member("faro_1963/1963-12-15_p23_sn83045462_0028_1963121501_0333.json") == 23
    assert page_from_member("weird.json") is None


def test_parse_scan_filters_and_legibility():
    scan = {
        "lccn": {"title": "Evening star.", "lccn": "sn1", "state": "DC"},
        "edition": {"date": "1930-05-16", "edition": "ed-01"},
        "page_number": "na", "scan_url": "u",
        "bboxes": [{"id": 1, "legibility": "Legible"}, {"id": 2, "legibility": "Illegible"}],
        "full articles": [
            {"object_ids": [1], "article": "x" * 300, "headline": "A", "byline": "", "full_article_id": 1},
            {"object_ids": [2], "article": "y" * 300, "headline": "B", "byline": "", "full_article_id": 2},
            {"object_ids": [1], "article": "short", "headline": "C", "byline": "", "full_article_id": 3},
        ],
    }
    import json
    arts = parse_scan("1930-05-16_p7_sn1_x.json", json.dumps(scan).encode(),
                      {"min_chars": 200, "max_chars": 20000, "keep_legibility": ["Legible", "Questionable", "NA"]},
                      "CC-BY-4.0")
    assert len(arts) == 1
    a = arts[0]
    assert a.page_number == 7 and a.year == 1930 and a.publication == "evening star" and a.legibility == "Legible"


def test_exact_and_near_duplicates_grouped():
    base = ("The county grand jury yesterday investigated the alleged attempt of a college student to "
            "extort four thousand dollars from a widow living on the east side of the city last week.")
    recs = [("a", base), ("b", base.upper() + "!!"), ("c", base.replace("yesterday", "today")),
            ("d", "An entirely different story about the baseball season opening in Washington next spring.")]
    g = dedupe(recs, threshold=0.5)
    assert g["a"]["exact_group"] == g["b"]["exact_group"]
    assert g["a"]["near_group"] == g["c"]["near_group"]
    assert g["d"]["near_group"] == "d"
    flags = group_flags(g, {"a": (1,), "b": (2,), "c": (3,), "d": (4,)})
    assert flags["a"]["near_dup"] is False and flags["c"]["near_dup"] is True and flags["b"]["exact_dup"] is True
    assert flags["a"]["near_size"] == 3


def test_small_helpers():
    assert exact_key("A  b") == exact_key("a b")
    assert shingles("one two", k=5) == {b"one two"}
    assert math.isclose(hhi([1, 1]), 0.5)
    assert junk_rate("ab~~") == 0.5
