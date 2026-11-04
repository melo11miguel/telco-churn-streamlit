// Reproduce en JavaScript el pipeline de scikit-learn exportado por
// export_web.py: imputación, estandarización, one-hot, suma ponderada y
// sigmoide. Funciona en el navegador (window.ChurnModel) y en Node (require).
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ChurnModel = api;
})(typeof self !== "undefined" ? self : this, function () {
  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  function isMissing(v) {
    return v === undefined || v === null || v === "" || Number.isNaN(v);
  }

  // Aporte de cada variable al logit, medido contra el cliente promedio de
  // entrenamiento: logit = base + suma de aportes.
  function contributions(model, input) {
    const parts = [];
    let base = model.intercept;

    for (const f of model.numeric) {
      const raw = isMissing(input[f.name]) ? f.impute : Number(input[f.name]);
      const z = (raw - f.mean) / f.scale;
      base += f.coef * f.train_mean;
      parts.push({ name: f.name, value: f.coef * (z - f.train_mean) });
    }

    for (const f of model.categorical) {
      const raw = isMissing(input[f.name]) ? f.impute : String(input[f.name]);
      let value = 0;
      f.levels.forEach((level, j) => {
        // handle_unknown="ignore": una categoría desconocida deja todo en 0.
        const x = level === raw ? 1 : 0;
        base += f.coef[j] * f.train_mean[j];
        value += f.coef[j] * (x - f.train_mean[j]);
      });
      parts.push({ name: f.name, value });
    }

    return { base, parts };
  }

  function logit(model, input) {
    const { base, parts } = contributions(model, input);
    return parts.reduce((acc, p) => acc + p.value, base);
  }

  function predict(model, input) {
    return sigmoid(logit(model, input));
  }

  // Mismas reglas que app.py: sin teléfono o sin internet, los servicios
  // dependientes toman el valor que siempre tienen en los datos.
  const DEPENDENCIES = {
    MultipleLines: ["PhoneService", "No phone service"],
    OnlineSecurity: ["InternetService", "No internet service"],
    OnlineBackup: ["InternetService", "No internet service"],
    DeviceProtection: ["InternetService", "No internet service"],
    TechSupport: ["InternetService", "No internet service"],
    StreamingTV: ["InternetService", "No internet service"],
    StreamingMovies: ["InternetService", "No internet service"],
  };

  function applyDependencies(values) {
    const out = { ...values };
    for (const [feature, [parent, fixed]] of Object.entries(DEPENDENCIES)) {
      if (out[parent] === "No") out[feature] = fixed;
      else if (out[feature] === fixed) out[feature] = "No";
    }
    return out;
  }

  // Cliente típico: mediana de las numéricas y moda de las categóricas.
  function defaults(model) {
    const values = {};
    for (const f of model.numeric) values[f.name] = f.median;
    for (const f of model.categorical) values[f.name] = f.default;
    return values;
  }

  // Los mismos ejemplos que tests/test_app.py y el README.
  const PRESETS = {
    tipico: {},
    alto: {
      tenure: 1,
      MonthlyCharges: 95,
      Contract: "Month-to-month",
      InternetService: "Fiber optic",
      PaymentMethod: "Electronic check",
      OnlineSecurity: "No",
      TechSupport: "No",
      PaperlessBilling: "Yes",
    },
    bajo: {
      tenure: 70,
      MonthlyCharges: 25,
      Contract: "Two year",
      InternetService: "No",
      PaymentMethod: "Bank transfer (automatic)",
      PaperlessBilling: "No",
    },
  };

  function preset(model, name) {
    return applyDependencies({ ...defaults(model), ...PRESETS[name] });
  }

  return {
    predict,
    logit,
    contributions,
    sigmoid,
    DEPENDENCIES,
    applyDependencies,
    defaults,
    PRESETS,
    preset,
  };
});
