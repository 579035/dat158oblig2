"""Data loading, species recovery, model building and the three-way decision rule."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from mushroom.schema import COLUMNS, FEATURES, UNKNOWN

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "agaricus-lepiota.data"
DROPPED = ["veil-type"]  # constant in the dataset
MODEL_FEATURES = [c for c in FEATURES if c not in DROPPED]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, header=None, names=COLUMNS, dtype=str)
    return df[MODEL_FEATURES], (df["class"] == "p").astype(int).to_numpy()


def distances(a, b):
    """Features differing between each row of a and its closest row in b, ignoring unknowns in a."""
    a, b = np.asarray(a), np.asarray(b)
    known = a != UNKNOWN
    out = np.empty(len(a), dtype=int)
    for i, (row, mask) in enumerate(zip(a, known)):
        out[i] = (b[:, mask] != row[mask]).sum(axis=1).min()
    return out


def species_groups(X):
    """Recover species: connected components of rows that differ in exactly one feature.

    The file has no species column, but this yields exactly the 23 species its
    documentation describes, each with a single edibility label.
    """
    arr = X.to_numpy()
    rows, cols = [], []
    for i in range(len(arr)):
        near = np.flatnonzero((arr[i + 1:] != arr[i]).sum(axis=1) <= 1) + i + 1
        rows.extend([i] * len(near))
        cols.extend(near)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(arr), len(arr)))
    return connected_components(graph, directed=False)[1]


def mask(X, rate, rng):
    """Replace a random share of cells by UNKNOWN."""
    arr = X.to_numpy(copy=True)
    arr[rng.random(arr.shape) < rate] = UNKNOWN
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def augment(X, y, rng, copies=4):
    """Add copies with random cells hidden so the model learns to work from partial input."""
    parts = [X] + [mask(X, rng.uniform(0.1, 0.8), rng) for _ in range(copies)]
    return pd.concat(parts, ignore_index=True), np.tile(y, copies + 1)


def learn_one_rule(X, y):
    """OneR: the single feature whose value-by-value majority vote fits the rows best."""
    best = None
    for column in X:
        votes = pd.Series(y, index=X.index).groupby(X[column]).mean().ge(0.5).astype(int)
        accuracy = (X[column].map(votes).to_numpy() == y).mean()
        if best is None or accuracy > best[0]:
            best = (accuracy, column, votes.to_dict())
    return best[1], best[2]


def rule_flags(rule, X):
    """True where the rule points to poison. A value it never saw counts; "Not sure" does not."""
    column, votes = rule
    return (X[column].ne(UNKNOWN) & X[column].map(votes).fillna(1).eq(1)).to_numpy()


def _to_str(X):
    return X.astype(str)


def candidates():
    def encoded(clf, dense=False):
        return make_pipeline(
            FunctionTransformer(_to_str),
            OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
            clf,
        )

    return {
        "baseline (majority)": encoded(DummyClassifier(strategy="most_frequent")),
        "naive bayes": encoded(BernoulliNB()),
        "logistic regression": encoded(LogisticRegression(C=1.0, max_iter=2000)),
        "decision tree": encoded(DecisionTreeClassifier(min_samples_leaf=5, random_state=0)),
        "random forest": encoded(RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=0)),
        "gradient boosting": encoded(HistGradientBoostingClassifier(max_iter=100, random_state=0), dense=True),
    }


@dataclass(frozen=True)
class Policy:
    """When the model is allowed to say edible. Everything else is a do-not-eat answer."""
    max_p_poison: float  # model must be at least this sure
    max_distance: int  # input must be this close to a mushroom seen in training
    min_known: int  # and enough features must be filled in

    def decide(self, p_poison, distance, n_known, rule_flag):
        edible = ~rule_flag & (p_poison <= self.max_p_poison) & (distance <= self.max_distance) & (n_known >= self.min_known)
        poisonous = (rule_flag | (p_poison >= 0.5)) & ~edible
        return np.where(edible, "edible", np.where(poisonous, "poisonous", "uncertain"))
