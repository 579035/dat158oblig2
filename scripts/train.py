"""Select, evaluate and fit the final model. Run: python -m scripts.train

Evaluation is built around the 23 recovered species:
  1. Five species are locked away as the final test set.
  2. On the other 18, candidates are compared with leave-one-species-out CV.
  3. The decision policy is tuned on the same 18 species.
  4. The chosen model and policy are scored once on the locked species.
  5. The final model is refit on everything and saved.
"""
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss, recall_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

from mushroom.model import (
    MODEL_FEATURES, Policy, augment, candidates, distances, learn_one_rule, load_data, mask,
    rule_flags, species_groups,
)
from mushroom.schema import UNKNOWN

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
SEED = 42
MASK_RATES = [0.0, 0.25, 0.5]
TEST_SPECIES = 5
MAX_FALSE_EDIBLE = 0.01
GRID = {
    "max_p_poison": [0.01, 0.02, 0.05, 0.1, 0.2],
    "max_distance": [0, 1, 2, 3, 4],
    "min_known": [6, 10, 14, 18],
}


def fit(model, X, y, seed=SEED):
    return clone(model).fit(*augment(X, y, np.random.default_rng(seed)))


def split_species(y, groups):
    """Pick test species with roughly the dataset's edible/poisonous mix."""
    rng = np.random.default_rng(SEED)
    labels = {g: y[groups == g][0] for g in np.unique(groups)}
    edible = rng.permutation([g for g, v in labels.items() if v == 0])
    poisonous = rng.permutation([g for g, v in labels.items() if v == 1])
    n_poison = TEST_SPECIES // 2 + 1
    return np.isin(groups, [*edible[: TEST_SPECIES - n_poison], *poisonous[:n_poison]])


def compare(X, y, groups):
    """Leave-one-species-out vs the naive random split, for every candidate."""
    rows = {}
    train_i, test_i = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    for name, model in candidates().items():
        random_acc = accuracy_score(
            y[test_i], fit(model, X.iloc[train_i], y[train_i]).predict(X.iloc[test_i]))
        oof = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            oof[te] = fit(model, X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        rows[name] = {
            "random_split_accuracy": round(random_acc, 4),
            "unseen_species_accuracy": round(accuracy_score(y, oof >= 0.5), 4),
            "unseen_species_poison_recall": round(recall_score(y, oof >= 0.5), 4),
            "unseen_species_log_loss": round(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6), labels=[0, 1]), 4),
        }
        print(f"{name:22s}", rows[name], flush=True)
    return rows


def scored_rows(model, X_train, y_train, X_test, y_test, rate, seed):
    """Probabilities and familiarity for held-out rows, with `rate` of their cells hidden."""
    model = fit(model, X_train, y_train)
    rule = learn_one_rule(X_train, y_train)
    X_seen = mask(X_test, rate, np.random.default_rng(seed))
    return pd.DataFrame({
        "p": model.predict_proba(X_seen)[:, 1],
        "d": distances(X_seen, X_train),
        "known": (X_seen.to_numpy() != UNKNOWN).sum(axis=1),
        "rule": rule_flags(rule, X_seen),
        "y": y_test,
        "rate": rate,
    })


def collect_unseen(model, X, y, groups):
    rng = np.random.default_rng(SEED)
    parts = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        te = rng.choice(te, size=min(len(te), 150), replace=False)
        parts += [scored_rows(model, X.iloc[tr], y[tr], X.iloc[te], y[te], r, SEED + i)
                  for i, r in enumerate(MASK_RATES)]
    return pd.concat(parts, ignore_index=True)


def collect_known(model, X, y):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    return pd.concat([scored_rows(model, X.iloc[tr], y[tr], X.iloc[te], y[te], r, SEED + i)
                      for i, r in enumerate(MASK_RATES)], ignore_index=True)


def score_policy(policy, rows):
    verdict = policy.decide(rows["p"].to_numpy(), rows["d"].to_numpy(), rows["known"].to_numpy(), rows["rule"].to_numpy())
    poison, edible = rows["y"].to_numpy() == 1, rows["y"].to_numpy() == 0
    return {
        "false_edible_rate": float((verdict[poison] == "edible").mean()),
        "edible_coverage": float((verdict[edible] == "edible").mean()),
        "poison_flagged_poisonous": float((verdict[poison] == "poisonous").mean()),
        "edible_flagged_poisonous": float((verdict[edible] == "poisonous").mean()),
        "uncertain_rate": float((verdict == "uncertain").mean()),
    }


def tune(unseen, known):
    """Most useful policy (edible answers on known species) that stays safe on unseen ones."""
    best = None
    for values in itertools.product(*GRID.values()):
        policy = Policy(*values)
        # The bound must hold at every mask rate, not just on average.
        worst = max(score_policy(policy, unseen[unseen["rate"] == r])["false_edible_rate"] for r in MASK_RATES)
        coverage = score_policy(policy, known)["edible_coverage"]
        if worst <= MAX_FALSE_EDIBLE and (best is None or coverage > best[0]):
            best = (coverage, policy)
    if best is None:
        sys.exit(f"no policy reaches a false-edible rate of {MAX_FALSE_EDIBLE}")
    return best[1]


def by_rate(policy, rows):
    return {str(r): {k: round(v, 4) for k, v in score_policy(policy, rows[rows["rate"] == r]).items()}
            for r in MASK_RATES}


def main():
    X, y = load_data()
    groups = species_groups(X)
    assert len(np.unique(groups)) == 23
    assert all(len(set(y[groups == g])) == 1 for g in np.unique(groups))

    is_test = split_species(y, groups)
    X_dev, y_dev, g_dev = X[~is_test], y[~is_test], groups[~is_test]
    X_test, y_test = X[is_test], y[is_test]
    print(f"dev: {len(y_dev)} rows, {len(set(g_dev))} species | locked test: {len(y_test)} rows, {len(set(groups[is_test]))} species")

    comparison = compare(X_dev, y_dev, g_dev)
    # Log loss ranks the probabilities that the policy depends on.
    usable = {k: v for k, v in comparison.items() if not k.startswith("baseline")}
    best_name = min(usable, key=lambda k: usable[k]["unseen_species_log_loss"])
    print("selected:", best_name)
    model = candidates()[best_name]

    unseen = collect_unseen(model, X_dev, y_dev, g_dev)
    known = collect_known(model, X_dev, y_dev)
    policy = tune(unseen, known)
    print("policy:", policy)

    test = pd.concat([scored_rows(model, X_dev, y_dev, X_test, y_test, r, SEED + 100 + i)
                      for i, r in enumerate(MASK_RATES)], ignore_index=True)
    plain = test[test["rate"] == 0]
    # The odor rule was added after the locked species had been scored once, so they are no
    # longer independent. Holding out each of the 23 species gives the fullest post-hoc estimate.
    everything = collect_unseen(model, X, y, groups)
    metrics = {
        "rows": len(y),
        "species": 23,
        "dev_species": len(set(g_dev)),
        "test_species": TEST_SPECIES,
        "model": best_name,
        "policy": vars(policy),
        "model_comparison": comparison,
        "dev_unseen_species": by_rate(policy, unseen),
        "dev_known_species": by_rate(policy, known),
        "test_unseen_species": by_rate(policy, test),
        "all_unseen_species": by_rate(policy, everything),
        "test_unseen_species_model_accuracy": round(accuracy_score(plain["y"], plain["p"] >= 0.5), 4),
    }

    final = fit(model, X, y)
    if "n_jobs" in final[-1].get_params():
        final[-1].set_params(n_jobs=1)  # a web request scores a single row
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({
        "model": final,
        "policy": policy,
        "rule": learn_one_rule(X, y),
        "features": MODEL_FEATURES,
        "train_rows": X.to_numpy(),
        "train_labels": y,
        "train_species": groups,
    }, ARTIFACTS / "model.joblib", compress=3)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps({k: metrics[k] for k in metrics if k != "model_comparison"}, indent=2, default=float))


if __name__ == "__main__":
    main()
