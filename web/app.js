// Interfaz del simulador. Toda la matemática vive en churn-model.js.
(function () {
  const CM = window.ChurnModel;
  const THRESHOLD = 0.5;
  const LOW = 0.3;

  const LABELS = {
    tenure: "Antigüedad",
    MonthlyCharges: "Cargo mensual",
    SeniorCitizen: "Adulto mayor",
    Partner: "Tiene pareja",
    Dependents: "Personas a cargo",
    PhoneService: "Servicio telefónico",
    MultipleLines: "Varias líneas",
    InternetService: "Servicio de internet",
    OnlineSecurity: "Seguridad en línea",
    OnlineBackup: "Respaldo en línea",
    DeviceProtection: "Protección de dispositivos",
    TechSupport: "Soporte técnico",
    StreamingTV: "TV por streaming",
    StreamingMovies: "Películas por streaming",
    Contract: "Tipo de contrato",
    PaperlessBilling: "Factura electrónica",
    PaymentMethod: "Método de pago",
  };

  const VALUE_LABELS = {
    Yes: "Sí",
    No: "No",
    "No phone service": "Sin teléfono",
    "No internet service": "Sin internet",
    DSL: "DSL",
    "Fiber optic": "Fibra óptica",
    "Month-to-month": "Mes a mes",
    "One year": "Un año",
    "Two year": "Dos años",
    "Bank transfer (automatic)": "Transferencia bancaria automática",
    "Credit card (automatic)": "Tarjeta de crédito automática",
    "Electronic check": "Cheque electrónico",
    "Mailed check": "Cheque por correo",
  };

  // Orden en que se muestran las opciones de cada control.
  const ORDER = {
    Contract: ["Month-to-month", "One year", "Two year"],
    InternetService: ["Fiber optic", "DSL", "No"],
    PaymentMethod: [
      "Electronic check",
      "Mailed check",
      "Bank transfer (automatic)",
      "Credit card (automatic)",
    ],
  };
  const YES_NO = ["Yes", "No"];
  const ADDONS = [
    "OnlineSecurity",
    "TechSupport",
    "OnlineBackup",
    "DeviceProtection",
    "StreamingTV",
    "StreamingMovies",
  ];

  // Cambios que un equipo de retención puede ofrecer.
  const LEVERS = [
    { feature: "Contract", from: ["Month-to-month"], to: "One year", text: "Ofrecer contrato de un año" },
    { feature: "Contract", from: ["Month-to-month", "One year"], to: "Two year", text: "Ofrecer contrato de dos años" },
    { feature: "PaymentMethod", from: ["Electronic check", "Mailed check"], to: "Credit card (automatic)", text: "Pasar a pago automático con tarjeta" },
    { feature: "PaymentMethod", from: ["Electronic check", "Mailed check"], to: "Bank transfer (automatic)", text: "Pasar a débito bancario automático" },
    { feature: "TechSupport", from: ["No"], to: "Yes", text: "Incluir soporte técnico" },
    { feature: "OnlineSecurity", from: ["No"], to: "Yes", text: "Incluir seguridad en línea" },
    { feature: "OnlineBackup", from: ["No"], to: "Yes", text: "Incluir respaldo en línea" },
    { feature: "DeviceProtection", from: ["No"], to: "Yes", text: "Incluir protección de dispositivos" },
  ];

  const ICONS = {
    low: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5l3 3 7-7"/></svg>',
    mid: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M8 4v5"/><circle cx="8" cy="12" r="0.6" fill="currentColor"/></svg>',
    high: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2.5l6 11H2z"/><path d="M8 7v3"/></svg>',
  };

  const nf0 = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 0 });
  const nf2 = new Intl.NumberFormat("es-CO", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const $ = (sel) => document.querySelector(sel);

  let model;
  let values;
  const numericSpec = {};
  const levels = {};

  function label(v, feature) {
    if (feature === "InternetService" && v === "No") return "Sin internet";
    return VALUE_LABELS[v] || v;
  }

  function optionsFor(feature) {
    const all = levels[feature];
    const dep = CM.DEPENDENCIES[feature];
    const base = ORDER[feature] || (all.includes("Yes") ? YES_NO : all);
    return base.filter((v) => all.includes(v) && (!dep || v !== dep[1]));
  }

  function riskLevel(p) {
    if (p >= THRESHOLD) return { key: "high", text: "Riesgo alto" };
    if (p >= LOW) return { key: "mid", text: "Riesgo medio" };
    return { key: "low", text: "Riesgo bajo" };
  }

  // ---------- Construcción del formulario ----------

  function buildSegmented(container, feature) {
    container.innerHTML = "";
    const title = document.createElement("span");
    title.className = "label";
    title.id = `lbl-${feature}`;
    title.textContent = LABELS[feature];
    const seg = document.createElement("div");
    seg.className = "seg";
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-labelledby", title.id);
    for (const v of optionsFor(feature)) {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.value = v;
      b.textContent = label(v, feature);
      b.addEventListener("click", () => update({ [feature]: v }));
      seg.appendChild(b);
    }
    container.append(title, seg);
  }

  function buildSelect(container, feature) {
    container.innerHTML = "";
    const id = `sel-${feature}`;
    const title = document.createElement("label");
    title.className = "label";
    title.htmlFor = id;
    title.textContent = LABELS[feature];
    const sel = document.createElement("select");
    sel.id = id;
    for (const v of optionsFor(feature)) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = label(v);
      sel.appendChild(o);
    }
    sel.addEventListener("change", () => update({ [feature]: sel.value }));
    container.append(title, sel);
  }

  function buildToggles() {
    const box = $("#addons");
    box.innerHTML = "";
    for (const feature of ADDONS) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "toggle";
      b.dataset.feature = feature;
      b.setAttribute("role", "switch");
      b.innerHTML = `<span class="toggle-text">${LABELS[feature]}</span><span class="switch" aria-hidden="true"></span>`;
      b.addEventListener("click", () => {
        update({ [feature]: values[feature] === "Yes" ? "No" : "Yes" });
      });
      box.appendChild(b);
    }
  }

  function buildNumeric(feature) {
    const spec = numericSpec[feature];
    const range = document.getElementById(feature);
    const num = document.getElementById(`${feature}-num`);
    for (const el of [range, num]) {
      el.min = spec.min;
      el.max = spec.max;
    }
    range.addEventListener("input", () => update({ [feature]: Number(range.value) }));
    num.addEventListener("change", () => {
      let v = Number(String(num.value).replace(",", "."));
      if (!Number.isFinite(v)) v = values[feature];
      v = Math.min(spec.max, Math.max(spec.min, v));
      if (feature === "tenure") v = Math.round(v);
      update({ [feature]: v });
    });
  }

  // ---------- Sincronizar controles con el estado ----------

  function syncControls() {
    for (const feature of Object.keys(numericSpec)) {
      const v = values[feature];
      document.getElementById(feature).value = v;
      const num = document.getElementById(`${feature}-num`);
      if (document.activeElement !== num) {
        num.value = feature === "tenure" ? v : Number(v).toFixed(2);
      }
    }

    document.querySelectorAll("[data-field]").forEach((container) => {
      const feature = container.dataset.field;
      const dep = CM.DEPENDENCIES[feature];
      const locked = dep && values[dep[0]] === "No";
      container.classList.toggle("is-disabled", Boolean(locked));
      container.querySelectorAll("button[data-value]").forEach((b) => {
        b.setAttribute("aria-pressed", String(!locked && b.dataset.value === values[feature]));
        b.disabled = Boolean(locked);
      });
      const sel = container.querySelector("select");
      if (sel) sel.value = values[feature];
      const title = container.querySelector(".label");
      if (title) {
        title.textContent = locked ? `${LABELS[feature]} (sin servicio telefónico)` : LABELS[feature];
      }
    });

    const noInternet = values.InternetService === "No";
    document.querySelectorAll(".toggle").forEach((b) => {
      const feature = b.dataset.feature;
      b.disabled = noInternet;
      b.setAttribute("aria-checked", String(values[feature] === "Yes"));
      b.title = noInternet ? "Requiere servicio de internet" : "";
    });
  }

  // ---------- Resultado ----------

  function renderResult() {
    const p = CM.predict(model, values);
    const score = Math.round(p * 100);
    const risk = riskLevel(p);
    const verdict = p >= THRESHOLD ? "Se prevé que se vaya" : "Se prevé que se quede";

    $("#score").textContent = score;
    const badgeHtml = `${ICONS[risk.key]}<span>${risk.text}</span>`;
    const badge = $("#badge");
    badge.className = `badge ${risk.key}`;
    badge.innerHTML = badgeHtml;
    badge.title = verdict;
    $("#marker").style.left = `${Math.min(100, Math.max(0, p * 100))}%`;

    $("#mobile-score").textContent = score;
    const mb = $("#mobile-badge");
    mb.className = `badge small ${risk.key}`;
    mb.innerHTML = badgeHtml;

    const idx = Math.min(9, Math.floor(p * 10));
    const band = model.calibration[idx];
    const rate = band.n ? band.churned / band.n : 0;
    const small = band.n < 30 ? " Es un grupo pequeño, así que tómelo como referencia aproximada." : "";
    $("#reality").innerHTML =
      `<strong>${verdict}.</strong> De los <strong>${nf0.format(band.n)}</strong> clientes de prueba ` +
      `con puntaje entre ${band.from * 100} y ${band.to * 100}, <strong>${nf0.format(rate * 100)} %</strong> ` +
      `realmente canceló.${small}`;

    renderDrivers(p);
    renderActions(p);
    renderCalibration(idx);
  }

  function describe(feature) {
    const v = values[feature];
    if (feature === "tenure") return `${nf0.format(v)} ${v === 1 ? "mes" : "meses"}`;
    if (feature === "MonthlyCharges") return `US$ ${nf2.format(v)}`;
    return label(v, feature);
  }

  function renderDrivers(p) {
    const { base, parts } = CM.contributions(model, values);
    const logit = parts.reduce((acc, x) => acc + x.value, base);
    const top = parts
      .map((x) => ({ ...x, pts: Math.round((p - CM.sigmoid(logit - x.value)) * 100) }))
      .filter((x) => x.pts !== 0)
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 5);

    const list = $("#drivers");
    if (!top.length) {
      list.innerHTML = '<li class="empty">Este cliente se parece mucho al cliente promedio.</li>';
      return;
    }
    const maxAbs = Math.max(...top.map((x) => Math.abs(x.value)));
    list.innerHTML = top
      .map((x) => {
        const dir = x.value > 0 ? "up" : "down";
        // Máximo 28 % por lado: deja espacio para la etiqueta de puntos.
        const width = (Math.abs(x.value) / maxAbs) * 28;
        const tag = `${x.pts > 0 ? "+" : ""}${x.pts} pts`;
        const tagPos = dir === "up"
          ? `left: calc(${50 + width}% + 6px)`
          : `right: calc(${50 + width}% + 6px)`;
        const verb = dir === "up" ? "sube" : "baja";
        return (
          `<li class="driver">` +
          `<span class="driver-name">${LABELS[x.name]}<small>${describe(x.name)}</small></span>` +
          `<span class="driver-bar" role="img" aria-label="${verb} el riesgo ${Math.abs(x.pts)} puntos">` +
          `<span class="driver-fill ${dir}" style="width:${width}%"></span>` +
          `<span class="driver-tag" style="${tagPos}">${tag}</span></span></li>`
        );
      })
      .join("");
  }

  function renderActions(p) {
    const best = {};
    for (const lever of LEVERS) {
      if (!lever.from.includes(values[lever.feature])) continue;
      const alt = CM.applyDependencies({ ...values, [lever.feature]: lever.to });
      if (alt[lever.feature] !== lever.to) continue;
      const q = CM.predict(model, alt);
      const drop = Math.round(p * 100) - Math.round(q * 100);
      if (drop < 1) continue;
      if (!best[lever.feature] || drop > best[lever.feature].drop) {
        best[lever.feature] = { ...lever, q, drop };
      }
    }
    const items = Object.values(best).sort((a, b) => b.drop - a.drop).slice(0, 3);
    const list = $("#actions");
    if (!items.length) {
      list.innerHTML = '<li class="empty">No hay cambios de contrato, pago o servicios que bajen el puntaje de forma notable.</li>';
      return;
    }
    list.innerHTML = items
      .map((x) =>
        `<li class="action"><span>${x.text}</span>` +
        `<span class="action-score">Puntaje <strong>${Math.round(x.q * 100)}</strong>` +
        `<span class="action-delta">-${x.drop}</span></span></li>`
      )
      .join("");
  }

  // ---------- Confiabilidad ----------

  function renderStats() {
    const t = model.metrics.test;
    const stats = [
      { value: nf0.format(t.recall_churn * 100), unit: " de cada 100", title: "Fugas detectadas", text: "De los clientes que sí se fueron, el modelo los marcó como riesgo alto." },
      { value: nf0.format(t.precision_churn * 100), unit: " de cada 100", title: "Alertas acertadas", text: "De los clientes marcados como riesgo alto, los que realmente se fueron." },
      { value: t.roc_auc.toFixed(2).replace(".", ","), unit: "", title: "Capacidad de ordenar (ROC-AUC)", text: `Entre un cliente que se fue y uno que se quedó, el primero recibe más puntaje el ${nf0.format(t.roc_auc * 100)} % de las veces.` },
      { value: nf0.format(model.metrics.n_test), unit: "", title: "Clientes de prueba", text: "Separados antes de entrenar. El modelo nunca los vio." },
    ];
    $("#stats").innerHTML = stats
      .map((s) =>
        `<div class="card stat"><span class="stat-value">${s.value}<small>${s.unit}</small></span>` +
        `<span class="stat-title">${s.title}</span><span class="stat-text">${s.text}</span></div>`
      )
      .join("");
    $("#n-test").textContent = nf0.format(model.metrics.n_test);
    $("#sk-version").textContent = model.sklearn_version;
  }

  let currentBand = 0;

  function renderCalibration(current) {
    currentBand = current;
    // Se dibuja al ancho real del contenedor para que el texto mida 12 px.
    const W = Math.max(300, $("#calib").clientWidth);
    const H = 240, padL = 44, padR = 8, padT = 24, padB = 26;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const bw = plotW / model.calibration.length;
    const y = (v) => padT + plotH * (1 - v);

    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
    svg += '<g class="grid">';
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      svg += `<line x1="${padL}" x2="${W - padR}" y1="${y(t)}" y2="${y(t)}"></line>`;
      svg += `<text x="${padL - 8}" y="${y(t) + 4}" text-anchor="end">${t * 100} %</text>`;
    }
    svg += "</g>";

    let currentLabel = "";
    model.calibration.forEach((b, i) => {
      const rate = b.n ? b.churned / b.n : 0;
      const x = padL + i * bw + 1;
      const w = bw - 2;
      const top = y(rate);
      const h = Math.max(0, y(0) - top);
      const r = Math.min(4, h, w / 2);
      const path = h > 0
        ? `M${x},${y(0)} V${top + r} Q${x},${top} ${x + r},${top} H${x + w - r} Q${x + w},${top} ${x + w},${top + r} V${y(0)} Z`
        : "";
      const tip = `Puntaje de ${b.from * 100} a ${b.to * 100}: ${nf0.format(rate * 100)} % canceló (${b.churned} de ${b.n} clientes)`;
      svg += `<rect class="hit" x="${padL + i * bw}" y="${padT}" width="${bw}" height="${plotH}" tabindex="0" data-tip="${tip}" aria-label="${tip}"></rect>`;
      svg += `<path class="bar${i === current ? " current" : ""}" d="${path}"></path>`;
      if (i === current) {
        const last = model.calibration.length - 1;
        const anchor = i === 0 ? "start" : i === last ? "end" : "middle";
        const lx = i === 0 ? x : i === last ? x + w : x + w / 2;
        currentLabel = `<text class="label" x="${lx}" y="${top - 8}" text-anchor="${anchor}">Su cliente: ${nf0.format(rate * 100)} %</text>`;
      }
    });
    // La etiqueta va al final para que ninguna barra vecina la tape.
    svg += currentLabel;

    svg += `<line class="baseline" x1="${padL}" x2="${W - padR}" y1="${y(0)}" y2="${y(0)}"></line>`;
    svg += '<g class="axis">';
    for (let i = 0; i <= model.calibration.length; i++) {
      svg += `<text x="${padL + i * bw}" y="${H - 6}" text-anchor="middle">${i * 10}</text>`;
    }
    svg += "</g></svg>";
    $("#calib").innerHTML = svg;
    bindTooltip();
  }

  function bindTooltip() {
    const tooltip = $("#tooltip");
    const card = $(".chart-card");
    const show = (el, clientX, clientY) => {
      tooltip.textContent = el.dataset.tip;
      tooltip.hidden = false;
      const box = card.getBoundingClientRect();
      const r = el.getBoundingClientRect();
      const x = clientX !== undefined ? clientX : r.left + r.width / 2;
      const yy = clientY !== undefined ? clientY : r.top + r.height / 3;
      const left = Math.min(box.width - tooltip.offsetWidth - 8, Math.max(8, x - box.left + 12));
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${yy - box.top - tooltip.offsetHeight - 10}px`;
    };
    const hide = () => { tooltip.hidden = true; };
    document.querySelectorAll("#calib .hit").forEach((el) => {
      el.addEventListener("mousemove", (e) => show(el, e.clientX, e.clientY));
      el.addEventListener("mouseleave", hide);
      el.addEventListener("focus", () => show(el));
      el.addEventListener("blur", hide);
    });
  }

  function renderTable() {
    const rows = model.calibration
      .map((b) => {
        const rate = b.n ? b.churned / b.n : 0;
        return `<tr><td>De ${b.from * 100} a ${b.to * 100}</td><td>${nf0.format(b.n)}</td><td>${nf0.format(b.churned)}</td><td>${nf0.format(rate * 100)} %</td></tr>`;
      })
      .join("");
    $("#calib-table").innerHTML =
      "<table><thead><tr><th>Puntaje</th><th>Clientes</th><th>Cancelaron</th><th>Fuga real</th></tr></thead>" +
      `<tbody>${rows}</tbody></table>`;
  }

  // ---------- Estado ----------

  function update(patch) {
    values = CM.applyDependencies({ ...values, ...patch });
    syncControls();
    renderResult();
  }

  function setupTheme() {
    const root = document.documentElement;
    const btn = $("#theme-toggle");
    const systemDark = window.matchMedia("(prefers-color-scheme: dark)");
    const isDark = () => root.dataset.theme === "dark" || (!root.dataset.theme && systemDark.matches);
    const paint = () => {
      btn.setAttribute("aria-label", isDark() ? "Cambiar a tema claro" : "Cambiar a tema oscuro");
      btn.dataset.mode = isDark() ? "dark" : "light";
    };
    btn.addEventListener("click", () => {
      root.dataset.theme = isDark() ? "light" : "dark";
      try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* sin almacenamiento */ }
      paint();
    });
    systemDark.addEventListener("change", paint);
    paint();
  }

  function setupMobileBar() {
    const bar = $("#mobile-bar");
    if (!("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver(
      (entries) => bar.classList.toggle("hide", entries[0].isIntersecting),
      { threshold: 0.25 }
    );
    io.observe($(".result"));
  }

  async function init() {
    model = await fetch("model.json").then((r) => r.json());
    for (const f of model.numeric) numericSpec[f.name] = f;
    for (const f of model.categorical) levels[f.name] = f.levels;

    document.querySelectorAll("[data-field]").forEach((container) => {
      const feature = container.dataset.field;
      if (feature === "PaymentMethod") buildSelect(container, feature);
      else buildSegmented(container, feature);
    });
    buildToggles();
    Object.keys(numericSpec).forEach(buildNumeric);

    document.querySelectorAll("[data-preset]").forEach((b) =>
      b.addEventListener("click", () => {
        values = CM.preset(model, b.dataset.preset);
        syncControls();
        renderResult();
      })
    );

    const toggle = $("#toggle-table");
    toggle.addEventListener("click", () => {
      const table = $("#calib-table");
      table.hidden = !table.hidden;
      toggle.setAttribute("aria-expanded", String(!table.hidden));
      toggle.textContent = table.hidden ? "Ver tabla" : "Ocultar tabla";
    });

    values = CM.preset(model, "tipico");
    renderStats();
    renderTable();
    syncControls();
    renderResult();
    setupMobileBar();

    let pending = 0;
    window.addEventListener("resize", () => {
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(() => renderCalibration(currentBand));
    });
  }

  setupTheme();

  init().catch((err) => {
    console.error(err);
    $("#reality").textContent = "No pudimos cargar el modelo. Recargue la página.";
  });
})();
