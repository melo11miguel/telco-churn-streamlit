"""Comprueba que la versión web usa el mismo modelo que model.joblib.

Si falla, hay que ejecutar:
    python export_web.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from export_web import export_model
from train import load_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def run_from_project_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_web_model_matches_joblib():
    exported = export_model(joblib.load(ROOT / "model.joblib"), load_data())
    on_disk = json.loads((ROOT / "web" / "model.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(exported))


def test_web_cases_match_joblib():
    model = joblib.load(ROOT / "model.joblib")["model"]
    cases = json.loads((ROOT / "tests" / "web_cases.json").read_text(encoding="utf-8"))
    X = pd.DataFrame([c["input"] for c in cases])
    proba = model.predict_proba(X)[:, 1]
    assert proba == pytest.approx([c["proba"] for c in cases], abs=1e-12)
