"""Extra measurements for the report. Run after training: python -m scripts.analysis

Species-level errors use only the 18 development species, as in model selection.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from mushroom.model import learn_one_rule, load_data, species_groups
from mushroom.predictor import MODEL_PATH, Predictor
from scripts.train import ARTIFACTS, candidates, fit, split_species


def feature_importance(bundle):
    pipeline = bundle["model"]
    names = pipeline[1].get_feature_names_out(bundle["features"])
    values = pd.Series(pipeline[-1].feature_importances_, index=names)
    grouped = values.groupby([n.rsplit("_", 1)[0] for n in names]).sum()
    return grouped.sort_values(ascending=False).round(4).to_dict()


def species_errors(X, y, groups, is_test):
    X_dev, y_dev, g_dev = X[~is_test], y[~is_test], groups[~is_test]
    model = candidates()["random forest"]
    rows = []
    for tr, te in LeaveOneGroupOut().split(X_dev, y_dev, g_dev):
        pred = fit(model, X_dev.iloc[tr], y_dev[tr]).predict_proba(X_dev.iloc[te])[:, 1] >= 0.5
        rows.append({
            "rows": len(te),
            "label": "poisonous" if y_dev[te][0] else "edible",
            "accuracy": round(float((pred == y_dev[te]).mean()), 4),
        })
    return sorted(rows, key=lambda r: r["accuracy"])


def odor_rule(X, y, is_test):
    """The simplest published rule (P_1 in the dataset documentation) on the same splits."""
    pred = (~X["odor"].isin(["a", "l", "n"])).to_numpy().astype(int)
    return {
        "all_rows_accuracy": round(float((pred == y).mean()), 4),
        "dev_species_accuracy": round(float((pred[~is_test] == y[~is_test]).mean()), 4),
        "test_species_accuracy": round(float((pred[is_test] == y[is_test]).mean()), 4),
        "test_species_false_edible": round(float((pred[is_test & (y == 1)] == 0).mean()), 4),
    }


def apply_one_rule(rule, X):
    column, votes = rule
    # A value absent from the training species gets the safe answer: poisonous.
    return X[column].map(votes).fillna(1).astype(int).to_numpy()


def one_rule(X, y, groups, is_test):
    """A fair version of odor_rule: learned only from training species, tested on others."""
    X_dev, y_dev, g_dev = X[~is_test], y[~is_test], groups[~is_test]
    oof, chosen = np.zeros(len(y_dev), dtype=int), []
    for tr, te in LeaveOneGroupOut().split(X_dev, y_dev, g_dev):
        rule = learn_one_rule(X_dev.iloc[tr], y_dev[tr])
        oof[te] = apply_one_rule(rule, X_dev.iloc[te])
        chosen.append(rule[0])
    rule = learn_one_rule(X_dev, y_dev)
    pred = apply_one_rule(rule, X[is_test])
    return {
        "features_chosen": pd.Series(chosen).value_counts().to_dict(),
        "dev_unseen_accuracy": round(float((oof == y_dev).mean()), 4),
        "dev_unseen_poison_recall": round(float(oof[y_dev == 1].mean()), 4),
        "test_rule": {"feature": rule[0], "poisonous_values": [k for k, v in rule[1].items() if v]},
        "test_accuracy": round(float((pred == y[is_test]).mean()), 4),
        "test_false_edible": round(float((pred[y[is_test] == 1] == 0).mean()), 4),
    }


def latency(X):
    predictor = Predictor()
    rows = [X.iloc[i].to_dict() for i in range(50)]
    times = []
    for row in rows:
        start = time.perf_counter()
        predictor.predict(row)
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": round(float(np.median(times)), 1), "max_ms": round(float(np.max(times)), 1)}


def main():
    X, y = load_data()
    groups = species_groups(X)
    is_test = split_species(y, groups)
    result = {
        "test_rows_poisonous": int(y[is_test].sum()),
        "test_rows_edible": int((1 - y[is_test]).sum()),
        "odor_rule": odor_rule(X, y, is_test),
        "one_rule": one_rule(X, y, groups, is_test),
        "feature_importance": feature_importance(joblib.load(MODEL_PATH)),
        "dev_species_errors": species_errors(X, y, groups, is_test),
        "latency": latency(X),
    }
    Path(ARTIFACTS / "analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
