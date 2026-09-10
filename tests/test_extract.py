"""Behavioural tests for the gender-signal hierarchy, entity resolution, roles and quotes.

These use the real spaCy small model on short constructed sentences (no corpus text),
so they check the rules as coded against parses the model gets right reliably.
"""
import pytest

spacy = pytest.importorskip("spacy")

from gna.extract import extract_article, resolve_gender  # noqa: E402
from gna.textnorm import normalize  # noqa: E402


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


def run(nlp, s):
    rows, art = extract_article(nlp(normalize(s)), {"article_id": "t", "year": 1900})
    return {r["name"]: r for r in rows}, art


# ---------------------------------------------------------------- hierarchy (pure function)
def test_honorific_beats_pronoun_and_is_recorded_for_consistency():
    g = resolve_gender("F", ["M", "M"], [])
    assert g["gender_hp"] == "F" and g["src_hp"] == "honorific" and g["pron_signal"] == "M"


def test_pronoun_requires_two_thirds_agreement():
    assert resolve_gender("N", ["F", "F", "M"], [])["gender_hp"] == "F"
    g = resolve_gender("N", ["F", "M"], [])
    assert g["gender_hp"] == "UNKNOWN" and g["src_hp"] == "conflict"


def test_unknown_is_default_and_honorific_only_tier_ignores_pronouns():
    g = resolve_gender("N", [None, None], [])
    assert g["gender_hp"] == "UNKNOWN" and g["src_hp"] == "none"
    assert resolve_gender("N", ["F"], [])["gender_h"] == "UNKNOWN"


def test_extended_tier_uses_gendered_nouns_before_pronouns():
    g = resolve_gender("N", ["M"], ["F"])
    assert g["gender_hpn"] == "F" and g["gender_hp"] == "M"


# ---------------------------------------------------------------- extraction on parses
def test_honorifics_assign_gender(nlp):
    ents, _ = run(nlp, "Mrs. Mary Jones spoke at the luncheon. Mr. Henry Brown said he would go.")
    assert ents["Mary Jones"]["gender_hp"] == "F"
    assert ents["Henry Brown"]["gender_hp"] == "M"


def test_pronoun_tier_only_in_hp(nlp):
    ents, _ = run(nlp, "Jane Doe arrived early. She was greeted by the committee.")
    e = ents["Jane Doe"]
    assert e["gender_h"] == "UNKNOWN" and e["gender_hp"] == "F" and e["src_hp"] == "pronoun"


def test_intervening_person_blocks_pronoun(nlp):
    ents, _ = run(nlp, "John Doe met Richard Roe and he left.")
    assert ents["John Doe"]["gender_hp"] == "UNKNOWN"


def test_wife_named_by_husband_is_not_merged_with_husband(nlp):
    # both entities carry the display name "Robert Jones", so inspect rows, not a name-keyed dict
    rows, _ = extract_article(nlp(normalize("Mrs. Robert Jones and Senator Robert Jones arrived at the station.")),
                              {"article_id": "t", "year": 1900})
    assert sorted(r["hclass"] for r in rows) == ["F", "N"]
    wife = next(r for r in rows if r["hclass"] == "F")
    husband = next(r for r in rows if r["hclass"] == "N")
    assert "PUBLIC_OFFICE" not in wife["roles_b"] and "PUBLIC_OFFICE" in husband["roles_b"]


def test_pronoun_in_next_sentence_goes_to_subject_not_embedded_pobj(nlp):
    rows, _ = extract_article(nlp(normalize(
        "Mrs. Mary Jones, wife of Senator Robert Jones, spoke at the luncheon. She said the club would meet.")),
        {"article_id": "t", "year": 1900})
    senator = next(r for r in rows if r["given"] == "robert")
    assert senator["gender_hp"] == "UNKNOWN" and senator["n_quotes_pron"] == 0


def test_dependency_method_does_not_inherit_husbands_office(nlp):
    ents, _ = run(nlp, "Mrs. Mary Jones, wife of Senator Robert Jones, spoke at the luncheon.")
    mary = ents["Mary Jones"]
    assert "FAMILY" in mary["roles_b"] and "PUBLIC_OFFICE" not in mary["roles_b"]
    assert "PUBLIC_OFFICE" in mary["roles_a"]          # the lexical window does inherit it


def test_org_head_disambiguated_by_organisation(nlp):
    ents, _ = run(nlp, "Mrs. Alice Hart was elected president of the Woman's Club.")
    hart = ents["Alice Hart"]
    assert "CIVIC" in hart["roles_b"] and "PUBLIC_OFFICE" not in hart["roles_b"]


def test_title_and_appositive_roles(nlp):
    ents, _ = run(nlp, "Gen. John J. Pershing said the army was ready.")
    assert "MILITARY" in ents["John J. Pershing"]["roles_b"]
    ents, _ = run(nlp, "Walter Johnson, the Washington pitcher, was injured yesterday.")
    assert {"ARTS_SPORTS", "CRIME_ACCIDENT"} <= set(ents["Walter Johnson"]["roles_b"])


def test_direct_quote_attribution(nlp):
    ents, _ = run(nlp, 'Mrs. Alice Hart spoke. "We will win the fight," Mrs. Hart said.')
    hart = ents["Alice Hart"]
    assert hart["n_quotes_name"] == 1 and hart["n_direct_name"] == 1 and hart["quote_chars_name"] > 10


def test_dr_and_mrs_does_not_give_wife_doctor_title(nlp):
    ents, _ = run(nlp, "Dr. and Mrs. John Smith attended the dinner.")
    smith = ents["John Smith"]
    assert smith["hclass"] == "F" and "PROFESSIONAL" not in smith["roles_b"]
