import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, predictor
from mushroom.model import Policy, distances, learn_one_rule, load_data, rule_flags, species_groups
from mushroom.schema import UNKNOWN

client = TestClient(app)
X, y = load_data()


def row(i):
    return X.iloc[i].to_dict()


def test_recovers_23_single_label_species():
    groups = species_groups(X)
    assert len(np.unique(groups)) == 23
    assert all(len(set(y[groups == g])) == 1 for g in np.unique(groups))


def test_distance_ignores_unknowns():
    train = pd.DataFrame([["a", "b", "c"]])
    assert distances([["a", "x", "z"]], train).tolist() == [2]
    assert distances([["a", UNKNOWN, "z"]], train).tolist() == [1]


def test_policy_only_says_edible_when_all_conditions_hold():
    policy = Policy(max_p_poison=0.02, max_distance=1, min_known=10)
    p, d, n = np.array([0.01, 0.03, 0.01, 0.01, 0.9]), np.array([1, 1, 2, 1, 5]), np.array([21, 21, 21, 9, 21])
    no_flag = np.zeros(5, dtype=bool)
    assert policy.decide(p, d, n, no_flag).tolist() == ["edible", "uncertain", "uncertain", "uncertain", "poisonous"]


def test_rule_flag_overrides_everything_but_never_unlocks_edible():
    policy = Policy(max_p_poison=0.02, max_distance=1, min_known=10)
    one = lambda v: np.array([v])
    assert policy.decide(one(0.01), one(0), one(21), one(True)).tolist() == ["poisonous"]
    assert policy.decide(one(0.3), one(5), one(3), one(False)).tolist() == ["uncertain"]


def test_rule_flags_unseen_values_but_not_unknown():
    rule = ("odor", {"f": 1, "n": 0})
    frame = pd.DataFrame({"odor": ["f", "n", UNKNOWN, "z"]})
    assert rule_flags(rule, frame).tolist() == [True, False, False, True]


def test_learned_rule_uses_odor():
    assert learn_one_rule(X, y)[0] == "odor"


def test_warning_odor_alone_says_poisonous_with_a_reason():
    body = client.post("/api/predict", json={"features": {"odor": "f"}}).json()
    assert body["verdict"] == "poisonous"
    assert "Foul" in body["reasons"][0]


def test_safe_odor_on_unfamiliar_mushroom_stays_uncertain_or_worse():
    body = client.post("/api/predict", json={"features": {"odor": "n"}}).json()
    assert body["verdict"] != "edible"


def test_every_training_row_is_answered_correctly_or_held_back():
    rows = X.to_numpy()
    p = predictor.model.predict_proba(X)[:, 1]
    verdict = predictor.policy.decide(p, np.zeros(len(rows), dtype=int), np.full(len(rows), rows.shape[1]), rule_flags(predictor.rule, X))
    assert not (verdict[y == 1] == "edible").any()


@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_predict_full_row(i):
    body = client.post("/api/predict", json={"features": row(i)}).json()
    assert body["verdict"] == ("poisonous" if y[i] else "edible")
    assert body["distance"] == 0
    assert len(body["factors"]) == 4


def test_empty_and_partial_input_never_say_edible():
    assert client.post("/api/predict", json={"features": {}}).json()["verdict"] != "edible"
    partial = client.post("/api/predict", json={"features": {"odor": "n"}}).json()
    assert partial["verdict"] != "edible"
    assert partial["n_known"] == 1


def test_value_absent_from_training_data_counts_as_unfamiliar():
    features = {**row(1), "gill-attachment": "n"}
    assert client.post("/api/predict", json={"features": features}).json()["distance"] >= 1


@pytest.mark.parametrize("features", [{"not-a-feature": "x"}, {"odor": "zz"}, {"veil-type": "p"}])
def test_invalid_input_is_rejected(features):
    assert client.post("/api/predict", json={"features": features}).status_code == 422


@pytest.mark.parametrize("kind,expected", [("edible", "edible"), ("poisonous", "poisonous")])
def test_examples(kind, expected):
    example = client.get(f"/api/example/{kind}").json()
    assert client.post("/api/predict", json={"features": example}).json()["verdict"] == expected


def test_unfamiliar_example_is_never_edible():
    for _ in range(20):
        example = client.get("/api/example/unfamiliar").json()
        assert client.post("/api/predict", json={"features": example}).json()["verdict"] != "edible"


def test_unknown_example_kind_rejected():
    assert client.get("/api/example/other").status_code == 422


def test_pages_and_metadata():
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert len(client.get("/api/schema").json()) == 21
    assert "model_comparison" in client.get("/api/metrics").json()
