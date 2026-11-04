// Prueba de la interfaz web: carga la página, pulsa botones y revisa el resultado.
//
// Uso:
//     npm install
//     npm test
//
// Con una URL prueba la versión publicada en lugar de la carpeta web/:
//     node tests/ui.test.cjs https://radar-fuga-telco.vercel.app/

const assert = require("node:assert");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { JSDOM } = require("jsdom");

const WEB = path.join(__dirname, "..", "web");
const TYPES = { ".html": "text/html", ".css": "text/css", ".js": "application/javascript", ".json": "application/json" };

function serveWeb() {
  const server = http.createServer((req, res) => {
    const file = path.join(WEB, decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/\/$/, "/index.html"));
    if (!file.startsWith(WEB) || !fs.existsSync(file)) {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

async function main() {
  let server;
  let base = process.argv[2];
  if (!base) {
    server = await serveWeb();
    base = `http://127.0.0.1:${server.address().port}/`;
  }

  const errors = [];
  const dom = await JSDOM.fromURL(base, {
    runScripts: "dangerously",
    resources: "usable",
    pretendToBeVisual: true,
    beforeParse(w) {
      w.matchMedia = () => ({ matches: false, addEventListener() {} });
      w.fetch = (u) => fetch(new URL(u, base));
      w.addEventListener("error", (e) => errors.push(e.message));
    },
  });
  const w = dom.window;
  const d = w.document;

  // Espera a que app.js cargue model.json y pinte el primer resultado.
  for (let i = 0; i < 50 && !d.querySelectorAll("#drivers li").length; i++) {
    await new Promise((r) => setTimeout(r, 100));
  }

  const score = () => Number(d.querySelector("#score").textContent);
  const click = (sel) => d.querySelector(sel).dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  const change = (el, value) => {
    el.value = value;
    el.dispatchEvent(new w.Event("change"));
  };

  // Ejemplos del README
  assert.strictEqual(score(), 72, "cliente típico");
  click('[data-preset="alto"]');
  assert.strictEqual(score(), 79, "alto riesgo");
  assert.match(d.querySelector("#badge").textContent, /Riesgo alto/);
  click('[data-preset="bajo"]');
  assert.strictEqual(score(), 1, "bajo riesgo");
  assert.match(d.querySelector("#badge").textContent, /Riesgo bajo/);

  // Sin internet, los servicios de internet quedan bloqueados
  assert.ok([...d.querySelectorAll(".toggle")].every((t) => t.disabled), "sin internet bloquea servicios");
  click('[data-field="InternetService"] button[data-value="DSL"]');
  assert.ok([...d.querySelectorAll(".toggle")].every((t) => !t.disabled), "con internet se habilitan");

  // Sin teléfono, varias líneas queda bloqueado
  click('[data-preset="tipico"]');
  click('[data-field="PhoneService"] button[data-value="No"]');
  const lines = [...d.querySelectorAll('[data-field="MultipleLines"] button')];
  assert.ok(lines.every((b) => b.disabled && b.getAttribute("aria-pressed") === "false"), "sin teléfono bloquea varias líneas");
  click('[data-field="PhoneService"] button[data-value="Yes"]');
  assert.ok(lines.every((b) => !b.disabled), "con teléfono se habilita");

  // Cambiar algo mueve el puntaje en la dirección esperada
  click('[data-preset="tipico"]');
  const before = score();
  click('.toggle[data-feature="TechSupport"]');
  assert.strictEqual(d.querySelector('.toggle[data-feature="TechSupport"]').getAttribute("aria-checked"), "true");
  assert.ok(score() < before, "soporte técnico baja el riesgo");
  click('[data-field="Contract"] button[data-value="Two year"]');
  assert.ok(score() < 50, "contrato de dos años saca al cliente de riesgo alto");

  // Los campos numéricos respetan el rango de los datos
  change(d.querySelector("#tenure-num"), "999");
  assert.strictEqual(d.querySelector("#tenure").value, "72");
  change(d.querySelector("#tenure-num"), "-5");
  assert.strictEqual(d.querySelector("#tenure").value, "0");
  change(d.querySelector("#sel-PaymentMethod"), "Mailed check");
  assert.strictEqual(d.querySelector("#sel-PaymentMethod").value, "Mailed check");

  // Explicación, acciones y confiabilidad
  click('[data-preset="alto"]');
  assert.strictEqual(d.querySelectorAll("#drivers li").length, 5);
  assert.ok(d.querySelectorAll("#actions .action").length >= 1, "propone al menos una acción");
  assert.strictEqual(d.querySelectorAll("#calib .bar").length, 10);
  assert.strictEqual(d.querySelectorAll("#calib .bar.current").length, 1);
  assert.strictEqual(d.querySelectorAll("#calib-table tbody tr").length, 10);
  assert.strictEqual(d.querySelectorAll("#stats .stat").length, 4);
  click("#toggle-table");
  assert.strictEqual(d.querySelector("#calib-table").hidden, false);
  click("#theme-toggle");
  assert.strictEqual(d.documentElement.dataset.theme, "dark");

  assert.deepStrictEqual(errors, [], "sin errores de JavaScript");
  console.log(`Interfaz OK (${base})`);
  w.close();
  if (server) server.close();
}

main().catch((err) => {
  console.error("FALLA:", err.message);
  process.exit(1);
});
