import numpy as np

from gna.llm import TEMPLATE, parse_reply, prompt_hash
from gna.roles_ml import BowRoleModel, bow_text, crossfit, roles_matrix


def test_bow_text_masks_the_name_and_marks_neighbours():
    ctx = "Mrs. Mary Jones, wife of Senator Robert Jones, spoke."
    t = bow_text(ctx, 5, 15)
    assert "mary" not in t.split()[:3] and "tgtperson" in t
    assert "l_mrs" in t and "r_wife" in t


def test_roles_matrix_ignores_unknown_roles():
    Y = roles_matrix([["CIVIC", "FAMILY"], [], ["NOT_A_ROLE"]])
    assert Y.shape == (3, 10) and Y.sum() == 2


def test_crossfit_is_out_of_fold_and_handles_rare_roles():
    rng = np.random.default_rng(0)
    words_pos = ["senator", "mayor", "judge", "governor"]
    texts, labels = [], []
    for i in range(80):
        pos = i % 2 == 0
        w = rng.choice(words_pos) if pos else rng.choice(["cake", "garden", "weather", "street"])
        texts.append(f"l_{w} tgtperson spoke {w} today")
        labels.append(["PUBLIC_OFFICE"] if pos else [])
    Y = roles_matrix(labels)
    P = crossfit(BowRoleModel, texts, Y, np.ones(len(Y)), k=4, seed=1)
    po = P[:, 0]
    assert po[Y[:, 0] == 1].mean() > 0.7 and po[Y[:, 0] == 0].mean() < 0.3
    # a role with no positives falls back to a constant prevalence predictor (0 here)
    assert np.allclose(P[:, 9], 0.0)


def test_parse_reply_is_strict_about_schema():
    raw = 'Sure! {"is_person": true, "gender_text": "f", "roles": ["CIVIC", "BOGUS"], "quoted": false, "confidence": 0.8}'
    d = parse_reply(raw)
    assert d == {"is_person": True, "gender_text": "F", "roles": ["CIVIC"], "quoted": False, "confidence": 0.8}
    assert parse_reply("no json here") is None
    assert parse_reply('{"gender_text": "nonbinary"}')["gender_text"] == "UNKNOWN"


def test_prompt_hash_is_stable_and_template_forbids_first_names():
    assert prompt_hash("m") == prompt_hash("m") and prompt_hash("m") != prompt_hash("n")
    assert "Do NOT use first names" in TEMPLATE
