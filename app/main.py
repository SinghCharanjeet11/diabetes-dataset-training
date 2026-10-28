from pathlib import Path
from typing import Literal
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
model = joblib.load(MODEL_DIR / "diabetes_model.joblib")
params = json.loads((MODEL_DIR / "model_params.json").read_text())

app = FastAPI(title="Diabetes Prediction API")


class PatientData(BaseModel):
    gender: Literal["Female", "Male", "Other"]
    age: float = Field(..., ge=0, le=120)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    smoking_history: Literal["never", "No Info", "current", "former", "ever", "not current"]
    bmi: float = Field(..., ge=10, le=100)
    HbA1c_level: float = Field(..., ge=3, le=15)
    blood_glucose_level: float = Field(..., ge=50, le=400)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return params


@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.model_dump()])[params["features"]]
    return {
        "prediction": int(model.predict(df)[0]),
        "probability": round(float(model.predict_proba(df)[0][1]), 4),
    }