"""Entrena el modelo de fuga de clientes (churn) y guarda model.joblib.

Uso:
    python train.py

Lee data/telco_churn.csv, compara dos modelos con validación cruzada sobre el
conjunto de entrenamiento, evalúa el elegido una sola vez sobre el conjunto de
prueba y guarda el pipeline completo (preprocesamiento + modelo) con sus
metadatos en model.joblib. Las métricas quedan también en metrics.json.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/telco_churn.csv")
MODEL_PATH = Path("model.joblib")
METRICS_PATH = Path("metrics.json")

RANDOM_STATE = 42
TARGET = "Churn"
POSITIVE_LABEL = "Yes"

# customerID es un identificador, gender no aporta señal y TotalCharges es
# redundante con tenure x MonthlyCharges (ver README).
numeric_features = ["tenure", "MonthlyCharges"]
categorical_features = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
features = numeric_features + categorical_features


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # SeniorCitizen viene como 0/1; lo dejamos como Yes/No igual que las demás.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def build_pipeline(classifier) -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def candidate_models() -> dict:
    # class_weight="balanced" compensa que solo ~26 % de los clientes se van.
    return {
        "Regresión logística": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def train_model(df: pd.DataFrame) -> dict:
    # 1. Definir variables
    X = df[features].copy()
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 2 y 3. Construir preprocesamiento y elegir modelo con validación cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, classifier in candidate_models().items():
        scores = cross_val_score(
            build_pipeline(classifier), X_train, y_train, cv=cv, scoring="f1"
        )
        cv_results[name] = {"f1_mean": scores.mean(), "f1_std": scores.std()}
        print(f"CV {name}: F1 = {scores.mean():.3f} +/- {scores.std():.3f}")

    # Regla de parsimonia: los candidatos van del más simple al más complejo y
    # nos quedamos con el primero que esté a menos de una desviación estándar
    # del mejor F1, porque una diferencia menor que eso es ruido de los folds.
    top_name = max(cv_results, key=lambda name: cv_results[name]["f1_mean"])
    top = cv_results[top_name]
    best_name = next(
        name for name, res in cv_results.items()
        if res["f1_mean"] >= top["f1_mean"] - top["f1_std"]
    )
    print(f"Modelo elegido: {best_name}")

    model = build_pipeline(candidate_models()[best_name])
    model.fit(X_train, y_train)

    # 4. Evaluar una sola vez sobre prueba
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    test_metrics = {
        "f1_churn": f1_score(y_test, pred),
        "recall_churn": recall_score(y_test, pred),
        "precision_churn": precision_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["Se queda", "Se va"]))
    print(f"ROC-AUC: {test_metrics['roc_auc']:.3f}")

    # 5. Construir artifact
    metrics = {
        "model_name": best_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "churn_rate": float(y.mean()),
        "cv_f1": cv_results,
        "test": test_metrics,
    }
    category_options = {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in categorical_features
    }
    category_defaults = {
        col: str(df[col].mode().iloc[0]) for col in categorical_features
    }
    numeric_ranges = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }
        for col in numeric_features
    }
    return {
        "model": model,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "category_options": category_options,
        "category_defaults": category_defaults,
        "numeric_ranges": numeric_ranges,
        "task": "classification",
        "classes": {0: "Se queda", 1: "Se va"},
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
    }


def main() -> None:
    df = load_data()
    artifact = train_model(df)
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(artifact["metrics"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
