"""Exporta el modelo entrenado a JSON para la versión web (carpeta web/).

Uso:
    python train.py
    python export_web.py

La regresión logística es una suma ponderada, así que el navegador puede
reproducir exactamente el pipeline de scikit-learn con estos parámetros:
estandarizar las numéricas, codificar one-hot las categóricas, multiplicar por
los coeficientes y aplicar la sigmoide. tests/check_web.mjs verifica que el
resultado coincide con model.predict_proba en los casos de tests/web_cases.json.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from train import (
    MODEL_PATH,
    POSITIVE_LABEL,
    RANDOM_STATE,
    TARGET,
    load_data,
)

WEB_MODEL_PATH = Path("web/model.json")
CASES_PATH = Path("tests/web_cases.json")
N_CASES = 300


def export_model(artifact: dict, df: pd.DataFrame) -> dict:
    model = artifact["model"]
    classifier = model.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression):
        raise SystemExit(
            "La versión web solo sabe reproducir una regresión logística; "
            f"train.py eligió {type(classifier).__name__}."
        )

    pre = model.named_steps["preprocessor"]
    num = pre.named_transformers_["num"]
    cat = pre.named_transformers_["cat"]
    coef = classifier.coef_[0]

    features = artifact["features"]
    X = df[features]
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    X_train, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Promedio de cada columna transformada en entrenamiento: es el "cliente
    # promedio" contra el que la web explica qué sube o baja el riesgo.
    train_means = np.asarray(pre.transform(X_train).mean(axis=0)).ravel()

    numeric = []
    for i, col in enumerate(artifact["numeric_features"]):
        numeric.append({
            "name": col,
            "impute": float(num.named_steps["imputer"].statistics_[i]),
            "mean": float(num.named_steps["scaler"].mean_[i]),
            "scale": float(num.named_steps["scaler"].scale_[i]),
            "coef": float(coef[i]),
            "train_mean": float(train_means[i]),
            "min": artifact["numeric_ranges"][col]["min"],
            "max": artifact["numeric_ranges"][col]["max"],
            "median": artifact["numeric_ranges"][col]["median"],
        })

    categorical = []
    offset = len(numeric)
    onehot = cat.named_steps["onehot"]
    for i, col in enumerate(artifact["categorical_features"]):
        levels = [str(v) for v in onehot.categories_[i]]
        span = slice(offset, offset + len(levels))
        categorical.append({
            "name": col,
            "impute": str(cat.named_steps["imputer"].statistics_[i]),
            "levels": levels,
            "coef": [float(c) for c in coef[span]],
            "train_mean": [float(m) for m in train_means[span]],
            "default": artifact["category_defaults"][col],
        })
        offset += len(levels)
    assert offset == len(coef)

    # Calibración: tasa real de fuga por tramo de puntaje, solo con clientes de
    # prueba que el modelo no vio al entrenar.
    test_proba = model.predict_proba(X_test)[:, 1]
    edges = np.linspace(0, 1, 11)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (test_proba >= lo) & ((test_proba < hi) | (hi == 1.0))
        bands.append({
            "from": round(float(lo), 1),
            "to": round(float(hi), 1),
            "n": int(mask.sum()),
            "churned": int(y_test[mask].sum()),
        })

    return {
        "model_name": artifact["metrics"]["model_name"],
        "sklearn_version": artifact["sklearn_version"],
        "intercept": float(classifier.intercept_[0]),
        "numeric": numeric,
        "categorical": categorical,
        "metrics": artifact["metrics"],
        "calibration": bands,
    }


def build_cases(artifact: dict, df: pd.DataFrame) -> list:
    """Clientes reales y combinaciones al azar con la probabilidad de sklearn."""
    rng = np.random.default_rng(RANDOM_STATE)
    features = artifact["features"]
    real = df[features].sample(N_CASES // 2, random_state=RANDOM_STATE)

    synthetic = {}
    for col in artifact["numeric_features"]:
        rng_col = artifact["numeric_ranges"][col]
        synthetic[col] = rng.uniform(rng_col["min"], rng_col["max"], N_CASES // 2)
    for col in artifact["categorical_features"]:
        synthetic[col] = rng.choice(artifact["category_options"][col], N_CASES // 2)
    cases = pd.concat([real, pd.DataFrame(synthetic)], ignore_index=True)

    proba = artifact["model"].predict_proba(cases[features])[:, 1]
    rows = cases.to_dict(orient="records")
    return [
        {"input": {k: (float(v) if k in artifact["numeric_features"] else str(v))
                   for k, v in row.items()},
         "proba": float(p)}
        for row, p in zip(rows, proba)
    ]


def main() -> None:
    artifact = joblib.load(MODEL_PATH)
    df = load_data()
    WEB_MODEL_PATH.parent.mkdir(exist_ok=True)
    WEB_MODEL_PATH.write_text(
        json.dumps(export_model(artifact, df), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    CASES_PATH.write_text(
        json.dumps(build_cases(artifact, df), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Modelo web guardado en {WEB_MODEL_PATH}")
    print(f"Casos de verificación guardados en {CASES_PATH}")


if __name__ == "__main__":
    main()
