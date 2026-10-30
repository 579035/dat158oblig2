"""Inference: turns form input into a verdict with its reasons."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mushroom.model import distances, rule_flags
from mushroom.schema import FEATURES, UNKNOWN

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model.joblib"
TOP_FACTORS = 4


class Predictor:
    def __init__(self, path=MODEL_PATH):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.policy = bundle["policy"]
        self.rule = bundle["rule"]
        self.features = bundle["features"]
        self.train_rows = bundle["train_rows"]
        self.train_labels = bundle["train_labels"]

    def _proba(self, rows):
        return self.model.predict_proba(pd.DataFrame(rows, columns=self.features))[:, 1]

    def predict(self, values):
        row = np.array([values.get(f, UNKNOWN) for f in self.features], dtype=object)
        known = row != UNKNOWN
        p_poison = float(self._proba(row[None, :])[0])
        distance = int(distances(row[None, :], self.train_rows)[0])
        n_known = int(known.sum())
        flag = rule_flags(self.rule, pd.DataFrame(row[None, :], columns=self.features))
        verdict = str(self.policy.decide(np.array([p_poison]), np.array([distance]), np.array([n_known]), flag)[0])

        reasons = []
        if flag[0]:
            label, options = FEATURES[self.rule[0]]
            reasons.append(f"{label}: {options[values[self.rule[0]]]} is a warning sign. In the data it only occurs in poisonous mushrooms.")
        if n_known < self.policy.min_known:
            reasons.append(f"Only {n_known} of {len(self.features)} features filled in; at least {self.policy.min_known} are needed before it can say edible.")
        if distance > self.policy.max_distance:
            reasons.append(f"The closest mushroom in the data differs in {distance} of the features you gave, so this may be a species the model has never seen.")
        if p_poison > self.policy.max_p_poison and verdict != "poisonous":
            reasons.append(f"The model's poison risk is {p_poison:.0%}, above the {self.policy.max_p_poison:.0%} limit for an edible answer.")

        return {
            "verdict": verdict,
            "p_poison": round(p_poison, 4),
            "distance": distance,
            "n_known": n_known,
            "n_features": len(self.features),
            "reasons": reasons,
            "factors": self._factors(row, known, p_poison),
        }

    def _factors(self, row, known, p_poison):
        """How much each given feature moves the risk: compare against hiding that feature."""
        idx = np.flatnonzero(known)
        if len(idx) == 0:
            return []
        hidden = np.tile(row, (len(idx), 1))
        hidden[np.arange(len(idx)), idx] = UNKNOWN
        effect = p_poison - self._proba(hidden)
        factors = []
        for i in np.argsort(-np.abs(effect))[:TOP_FACTORS]:
            name = self.features[idx[i]]
            label, options = FEATURES[name]
            factors.append({
                "feature": label,
                "value": options.get(row[idx[i]], row[idx[i]]),
                "effect": round(float(effect[i]), 4),
            })
        return factors

    def example(self, kind, rng):
        """A real row from the data, or for 'unfamiliar' one altered to resemble no known species."""
        if kind == "unfamiliar":
            for _ in range(200):
                row = self.train_rows[rng.integers(len(self.train_rows))].copy()
                for i in rng.choice(len(row), size=5, replace=False):
                    options = [c for c in FEATURES[self.features[i]][1] if c != row[i]]
                    row[i] = rng.choice(options)
                if distances(row[None, :], self.train_rows)[0] > self.policy.max_distance:
                    break
        else:
            pool = self.train_rows[self.train_labels == (kind == "poisonous")]
            row = pool[rng.integers(len(pool))]
        return dict(zip(self.features, map(str, row)))
