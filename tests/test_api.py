from pathlib import Path
import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
SAMPLE = json.loads((Path(__file__).resolve().parent.parent / "sample_request.json").read_text())


def test_health():
    assert client.get("/health").status_code == 200


def test_predict():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert r.json()["prediction"] in (0, 1)


def test_predict_invalid_input():
    assert client.post("/predict", json={**SAMPLE, "gender": "Unknown"}).status_code == 422