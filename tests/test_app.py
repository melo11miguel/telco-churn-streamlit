"""Pruebas mínimas antes de construir la imagen Docker.

Uso:
    python train.py
    pytest -q
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")

HIGH_RISK = {
    "tenure": 1,
    "MonthlyCharges": 95.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "PaperlessBilling": "Yes",
}
LOW_RISK = {
    "tenure": 70,
    "MonthlyCharges": 25.0,
    "Contract": "Two year",
    "InternetService": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "PaperlessBilling": "No",
}
INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


@pytest.fixture(autouse=True)
def run_from_project_root(monkeypatch):
    # app.py abre model.joblib con ruta relativa, como en el contenedor.
    monkeypatch.chdir(ROOT)


@pytest.fixture(scope="module")
def artifact():
    return joblib.load(ROOT / "model.joblib")


def run_case(case: dict) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    numbers = {"Antigüedad (meses)": "tenure", "Cargo mensual (USD)": "MonthlyCharges"}
    for widget in at.number_input:
        feature = numbers[widget.label]
        if feature in case:
            widget.set_value(case[feature])
    for widget in at.selectbox:
        if widget.key in case:
            widget.set_value(case[widget.key])
    at.button[0].click().run()
    assert not at.exception, at.exception
    return at


def churn_probability(at: AppTest) -> float:
    messages = [m.value for m in (*at.error, *at.success)]
    assert len(messages) == 1, messages
    return float(messages[0].rsplit(":", 1)[1].strip().rstrip("%")) / 100


def test_artifact_has_interface_metadata(artifact):
    assert artifact["task"] == "classification"
    assert artifact["features"] == (
        artifact["numeric_features"] + artifact["categorical_features"]
    )
    for col in artifact["categorical_features"]:
        assert artifact["category_options"][col]
        assert artifact["category_defaults"][col] in artifact["category_options"][col]


def test_app_loads_without_training():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert len(at.number_input) == 2
    assert len(at.selectbox) == 15
    assert not any(w.disabled for w in at.selectbox)


def test_default_case_runs():
    at = run_case({})
    assert 0.0 <= churn_probability(at) <= 1.0


def test_high_risk_case_is_flagged():
    at = run_case(HIGH_RISK)
    assert at.error, "un cliente nuevo, mes a mes y con fibra debería marcarse"
    assert churn_probability(at) > 0.7


def test_low_risk_case_stays():
    at = run_case(LOW_RISK)
    assert at.success
    assert churn_probability(at) < 0.1


def test_extreme_values_run():
    run_case({"tenure": 0, "MonthlyCharges": 18.25})
    run_case({"tenure": 72, "MonthlyCharges": 118.75})


def test_defaults_are_most_frequent_values(artifact):
    at = AppTest.from_file(APP, default_timeout=30).run()
    for widget in at.selectbox:
        assert widget.value == artifact["category_defaults"][widget.key]


def test_dependent_services_follow_parent():
    at = run_case({"InternetService": "No", "PhoneService": "No"})
    fixed = {w.key: w for w in at.selectbox if w.key.endswith("_fixed")}
    assert set(fixed) == {f"{f}_fixed" for f in INTERNET_ADDONS + ["MultipleLines"]}
    assert all(w.disabled for w in fixed.values())
    assert fixed["MultipleLines_fixed"].value == "No phone service"
    for feature in INTERNET_ADDONS:
        assert fixed[f"{feature}_fixed"].value == "No internet service"

    at = run_case({"InternetService": "DSL"})
    options = {w.key: w.options for w in at.selectbox}
    for feature in INTERNET_ADDONS:
        assert "Sin servicio de internet" not in options[feature]


def test_app_matches_pipeline(artifact):
    # HIGH_RISK tiene teléfono e internet, así que ninguna regla de
    # dependencia cambia los valores por defecto.
    at = run_case(HIGH_RISK)
    row = {}
    for col in artifact["numeric_features"]:
        row[col] = HIGH_RISK.get(col, artifact["numeric_ranges"][col]["median"])
    for col in artifact["categorical_features"]:
        row[col] = HIGH_RISK.get(col, artifact["category_defaults"][col])
    X = pd.DataFrame([row], columns=artifact["features"])
    expected = artifact["model"].predict_proba(X)[0, 1]
    assert churn_probability(at) == pytest.approx(expected, abs=0.0005)
