from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

BASE = Path(__file__).resolve().parent.parent


def test_model_accuracy():
    model = joblib.load(BASE / "model" / "diabetes_model.joblib")
    params = json.loads((BASE / "model" / "model_params.json").read_text())
    df = pd.read_csv(BASE / "data" / "test_data.csv")
    acc = accuracy_score(df[params["target"]], model.predict(df[params["features"]]))
    assert acc >= 0.90