"""Exploratory checks that shaped the modelling decisions. Run: python -m scripts.explore"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from mushroom.schema import COLUMNS

df = pd.read_csv(Path("data/raw/agaricus-lepiota.data"), header=None, names=COLUMNS, dtype=str)
X, y = df.drop(columns="class"), (df["class"] == "p").astype(int)

print("shape", df.shape, "| poisonous", y.mean().round(3), "| duplicates", df.duplicated().sum())
print("conflicting duplicates (same X, both labels):", X.duplicated(keep=False).sum() - df.duplicated(keep=False).sum())
print("constant columns:", [c for c in X if X[c].nunique() == 1])
print("missing:", (X == "?").sum().loc[lambda s: s > 0].to_dict())
print("stalk-root missing, by class:", df[df["stalk-root"] == "?"]["class"].value_counts().to_dict())

codes = X.apply(lambda c: c.astype("category").cat.codes).to_numpy()
dist = squareform(pdist(codes, metric="hamming")) * X.shape[1]
np.fill_diagonal(dist, np.inf)
print("nearest neighbour distance (features differing):", pd.Series(dist.min(axis=1)).value_counts().sort_index().to_dict())

model = make_pipeline(OneHotEncoder(handle_unknown="ignore"), DecisionTreeClassifier(random_state=0))
print("\nrandom 5-fold accuracy:", cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean().round(4))

for k in range(1, 7):
    n, groups = connected_components(dist <= k, directed=False)
    sizes = pd.Series(groups).value_counts()
    pure = df.groupby(groups)["class"].nunique().eq(1).mean()
    line = f"k={k}: {n} groups, largest {sizes.iloc[0]}, label-pure {pure:.2f}"
    if 5 <= n:
        s = cross_val_score(model, X, y, cv=GroupKFold(5), groups=groups)
        line += f", group 5-fold acc {s.mean():.4f} (min fold {s.min():.4f})"
    print(line)
