"""FastAPI backend. Run: uvicorn app.main:app --port 7860"""
import json
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mushroom.model import MODEL_FEATURES
from mushroom.predictor import MODEL_PATH, Predictor
from mushroom.schema import FEATURES, UNKNOWN

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Mushroom edibility classifier")
predictor = Predictor()
metrics = json.loads((MODEL_PATH.parent / "metrics.json").read_text())
rng = np.random.default_rng()


class PredictRequest(BaseModel):
    features: dict[str, str]


@app.get("/api/schema")
def schema():
    return [
        {"name": name, "label": FEATURES[name][0],
         "options": [{"code": c, "label": l} for c, l in FEATURES[name][1].items()]}
        for name in MODEL_FEATURES
    ]


@app.get("/api/metrics")
def get_metrics():
    return metrics


@app.get("/api/example/{kind}")
def example(kind: Literal["edible", "poisonous", "unfamiliar"]):
    return predictor.example(kind, rng)


@app.post("/api/predict")
def predict(request: PredictRequest):
    for name, code in request.features.items():
        if name not in MODEL_FEATURES:
            raise HTTPException(422, f"Unknown feature: {name}")
        if code != UNKNOWN and code not in FEATURES[name][1]:
            raise HTTPException(422, f"Invalid value for {name}: {code}")
    return predictor.predict(request.features)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
