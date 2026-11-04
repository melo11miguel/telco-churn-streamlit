"""Interfaz Streamlit para estimar la probabilidad de fuga de un cliente.

Uso:
    streamlit run app.py

Solo carga model.joblib; no vuelve a entrenar. Todas las transformaciones las
aplica el pipeline guardado, así que la app envía los valores tal como los
vio el modelo en entrenamiento.
"""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model.joblib"
THRESHOLD = 0.5

LABELS = {
    "tenure": "Antigüedad (meses)",
    "MonthlyCharges": "Cargo mensual (USD)",
    "SeniorCitizen": "Adulto mayor",
    "Partner": "Tiene pareja",
    "Dependents": "Tiene personas a cargo",
    "PhoneService": "Servicio telefónico",
    "MultipleLines": "Varias líneas",
    "InternetService": "Servicio de internet",
    "OnlineSecurity": "Seguridad en línea",
    "OnlineBackup": "Respaldo en línea",
    "DeviceProtection": "Protección de dispositivos",
    "TechSupport": "Soporte técnico",
    "StreamingTV": "TV por streaming",
    "StreamingMovies": "Películas por streaming",
    "Contract": "Tipo de contrato",
    "PaperlessBilling": "Factura electrónica",
    "PaymentMethod": "Método de pago",
}

# Solo cambia lo que se muestra; al modelo le llega el valor original.
VALUE_LABELS = {
    "Yes": "Sí",
    "No": "No",
    "No phone service": "Sin servicio telefónico",
    "No internet service": "Sin servicio de internet",
    "DSL": "DSL",
    "Fiber optic": "Fibra óptica",
    "Month-to-month": "Mes a mes",
    "One year": "Un año",
    "Two year": "Dos años",
    "Bank transfer (automatic)": "Transferencia bancaria (automática)",
    "Credit card (automatic)": "Tarjeta de crédito (automática)",
    "Electronic check": "Cheque electrónico",
    "Mailed check": "Cheque por correo",
}

GROUPS = {
    "Perfil del cliente": ["SeniorCitizen", "Partner", "Dependents"],
    "Contrato y facturación": [
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ],
    "Servicios contratados": [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ],
}

# En los datos, sin teléfono MultipleLines siempre es "No phone service" y sin
# internet estos seis servicios siempre son "No internet service". La app
# respeta esas mismas reglas para no pedir combinaciones que nunca existieron.
DEPENDENCIES = {
    "MultipleLines": ("PhoneService", "No phone service"),
    "OnlineSecurity": ("InternetService", "No internet service"),
    "OnlineBackup": ("InternetService", "No internet service"),
    "DeviceProtection": ("InternetService", "No internet service"),
    "TechSupport": ("InternetService", "No internet service"),
    "StreamingTV": ("InternetService", "No internet service"),
    "StreamingMovies": ("InternetService", "No internet service"),
}


st.set_page_config(page_title="Fuga de clientes Telco", page_icon="📉")


@st.cache_resource
def load_artifact(path: str = MODEL_PATH) -> dict:
    return joblib.load(path)


artifact = load_artifact()
model = artifact["model"]
features = artifact["features"]
numeric_features = artifact["numeric_features"]
categorical_features = artifact["categorical_features"]
category_options = artifact["category_options"]
category_defaults = artifact.get("category_defaults", {})
numeric_ranges = artifact["numeric_ranges"]
metrics = artifact["metrics"]

st.title("¿Este cliente se va a ir?")
st.write(
    "Ingrese las características de un cliente de telecomunicaciones y el "
    "modelo estimará la probabilidad de que cancele el servicio (churn)."
)

with st.sidebar:
    st.header("Sobre el modelo")
    test = metrics["test"]
    st.write(f"**Modelo:** {metrics['model_name']}")
    st.write(f"Entrenado con {metrics['n_train']:,} clientes y evaluado con {metrics['n_test']:,}.")
    st.metric("Recall de fuga (prueba)", f"{test['recall_churn']:.0%}")
    st.metric("F1 de fuga (prueba)", f"{test['f1_churn']:.2f}")
    st.metric("ROC-AUC (prueba)", f"{test['roc_auc']:.3f}")
    st.caption(
        f"En los datos, el {metrics['churn_rate']:.0%} de los clientes se fue. "
        f"Umbral de decisión: {THRESHOLD:.0%}."
    )

values = {}

st.subheader("Cuenta")
col1, col2 = st.columns(2)
for col, feature in zip((col1, col2), numeric_features):
    rng = numeric_ranges[feature]
    is_int = feature == "tenure"
    cast = int if is_int else float
    values[feature] = col.number_input(
        LABELS.get(feature, feature),
        min_value=cast(rng["min"]),
        max_value=cast(rng["max"]),
        value=cast(rng["median"]),
        step=1 if is_int else 1.0,
    )

grouped = [f for group in GROUPS.values() for f in group]
groups = dict(GROUPS)
leftover = [f for f in categorical_features if f not in grouped]
if leftover:
    groups["Otras variables"] = leftover

for title, group in groups.items():
    st.subheader(title)
    cols = st.columns(3)
    for i, feature in enumerate(f for f in group if f in categorical_features):
        options = category_options[feature]
        label = LABELS.get(feature, feature)
        parent, fixed = DEPENDENCIES.get(feature, (None, None))

        if parent and values.get(parent) == "No":
            values[feature] = cols[i % 3].selectbox(
                label,
                [fixed],
                format_func=lambda v: VALUE_LABELS.get(v, v),
                disabled=True,
                key=f"{feature}_fixed",
            )
            continue

        if parent:
            options = [o for o in options if o != fixed]
        default = category_defaults.get(feature)
        values[feature] = cols[i % 3].selectbox(
            label,
            options,
            index=options.index(default) if default in options else 0,
            format_func=lambda v: VALUE_LABELS.get(v, v),
            key=feature,
        )

if st.button("Ejecutar modelo", type="primary"):
    X_new = pd.DataFrame([values], columns=features)
    proba = float(model.predict_proba(X_new)[0, 1])
    leaves = proba >= THRESHOLD

    if leaves:
        st.error(f"Clase predicha: **Se va**. Probabilidad de fuga: {proba:.1%}")
    else:
        st.success(f"Clase predicha: **Se queda**. Probabilidad de fuga: {proba:.1%}")
    st.progress(proba)

    st.info(
        "Cómo leer el resultado: la probabilidad es la estimación del modelo "
        "para un cliente con este perfil. Si supera el umbral, lo marcamos "
        "como riesgo de fuga para priorizarlo en una campaña de retención. "
        f"El modelo detecta cerca del {metrics['test']['recall_churn']:.0%} "
        "de los clientes que realmente se van, a cambio de algunas falsas "
        "alarmas."
    )
