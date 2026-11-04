// Verifica que la versión web da la misma probabilidad que scikit-learn.
//
// Uso:
//     python export_web.py
//     node tests/check_web.mjs

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { predict, preset, contributions, sigmoid } = require("../web/churn-model.js");

const model = JSON.parse(readFileSync(new URL("../web/model.json", import.meta.url)));
const cases = JSON.parse(readFileSync(new URL("./web_cases.json", import.meta.url)));

const TOLERANCE = 1e-9;
let worst = 0;
for (const { input, proba } of cases) {
  worst = Math.max(worst, Math.abs(predict(model, input) - proba));
}

// Una categoría desconocida debe comportarse como handle_unknown="ignore".
const unknown = predict(model, { ...cases[0].input, Contract: "Ninguno" });
if (!(unknown > 0 && unknown < 1)) throw new Error("categoría desconocida rompe la predicción");

console.log(`${cases.length} casos, diferencia máxima con sklearn: ${worst.toExponential(2)}`);
if (worst > TOLERANCE) {
  console.error(`FALLA: la diferencia supera ${TOLERANCE}`);
  process.exit(1);
}

// Los ejemplos deben dar lo documentado en el README.
const expected = { tipico: 72.1, alto: 79.2, bajo: 1.3 };
for (const [name, pct] of Object.entries(expected)) {
  const got = Math.round(predict(model, preset(model, name)) * 1000) / 10;
  console.log(`ejemplo ${name}: ${got} % (README: ${pct} %)`);
  if (got !== pct) {
    console.error(`FALLA: el ejemplo ${name} no coincide con el README`);
    process.exit(1);
  }
}

// Base más aportes debe reconstruir exactamente el logit.
const { base, parts } = contributions(model, preset(model, "alto"));
const rebuilt = sigmoid(parts.reduce((acc, p) => acc + p.value, base));
if (Math.abs(rebuilt - predict(model, preset(model, "alto"))) > TOLERANCE) {
  console.error("FALLA: los aportes no suman el logit");
  process.exit(1);
}

const bajo = preset(model, "bajo");
if (bajo.OnlineSecurity !== "No internet service" || bajo.MultipleLines !== "No") {
  console.error("FALLA: las reglas de dependencia no se aplicaron");
  process.exit(1);
}
console.log("OK");
