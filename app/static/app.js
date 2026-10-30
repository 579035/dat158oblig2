const GROUPS = {
  "Cap": ["cap-shape", "cap-surface", "cap-color"],
  "Gills": ["gill-attachment", "gill-spacing", "gill-size", "gill-color"],
  "Stalk": ["stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
            "stalk-color-above-ring", "stalk-color-below-ring"],
  "Veil and ring": ["veil-color", "ring-number", "ring-type"],
  "Smell and touch": ["odor", "bruises"],
  "Spores and surroundings": ["spore-print-color", "population", "habitat"],
};

const VERDICTS = {
  edible: ["Looks edible", "It closely matches edible species in the data. This is not proof it is safe, so still don't eat it."],
  poisonous: ["Likely poisonous", "Do not eat."],
  uncertain: ["Can't tell", "Treat it as poisonous. The model doesn't know enough to say otherwise."],
};

const form = document.getElementById("form");
const result = document.getElementById("result");
let latestRequest = 0;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function buildForm(schema) {
  const byName = Object.fromEntries(schema.map(f => [f.name, f]));
  for (const [title, names] of Object.entries(GROUPS)) {
    const fieldset = el("fieldset");
    fieldset.append(el("legend", "", title));
    const fields = el("div", "fields");
    for (const name of names) {
      const label = el("label", "", byName[name].label);
      const select = el("select");
      select.name = name;
      select.append(new Option("Not sure", "?"));
      for (const o of byName[name].options) select.append(new Option(o.label, o.code));
      label.append(select);
      fields.append(label);
    }
    fieldset.append(fields);
    form.append(fieldset);
  }
}

function selects() {
  return [...form.querySelectorAll("select")];
}

function values() {
  return Object.fromEntries(selects().map(s => [s.name, s.value]));
}

function markSelected() {
  for (const s of selects()) s.classList.toggle("set", s.value !== "?");
}

function percent(p) {
  if (p > 0 && p < 0.01) return "<1%";
  return `${Math.round(p * 100)}%`;
}

function renderEmpty() {
  const panel = el("div", "panel");
  panel.append(el("p", "", "Your result appears here as soon as you fill in a feature."));
  result.replaceChildren(panel);
}

function renderError() {
  const panel = el("div", "panel");
  panel.append(el("p", "", "Something went wrong reaching the model. Please try again."));
  result.replaceChildren(panel);
}

function renderResult(r) {
  const panel = el("div", "panel");
  const [title, text] = VERDICTS[r.verdict];
  const verdict = el("div", `verdict ${r.verdict}`);
  verdict.append(el("h3", "", title), el("p", "", text));

  const stat = el("div", "stat");
  stat.append(el("span", "", "Model's poison risk"), el("b", "", percent(r.p_poison)));
  const meter = el("div", "meter");
  meter.setAttribute("role", "img");
  meter.setAttribute("aria-label", `Poison risk ${percent(r.p_poison)}`);
  const fill = el("i");
  fill.style.width = `${r.p_poison * 100}%`;
  meter.append(fill);

  const known = el("p", "note", `${r.n_known} of ${r.n_features} features given.`);
  panel.append(verdict, stat, meter, known);

  if (r.reasons.length) {
    panel.append(el("h4", "", "Why this answer"));
    const list = el("ul");
    for (const reason of r.reasons) list.append(el("li", "", reason));
    panel.append(list);
  }

  // Features can be redundant, so hiding one may change nothing; those explain nothing here.
  const factors = r.factors.filter(f => Math.round(f.effect * 100) !== 0);
  if (factors.length) {
    panel.append(el("h4", "", "What moved the risk most"));
    const max = Math.max(...factors.map(f => Math.abs(f.effect)));
    for (const f of factors) {
      const row = el("div", "factor");
      const change = Math.round(f.effect * 100);
      const direction = `${change > 0 ? "raises" : "lowers"} ${Math.abs(change)} pts`;
      row.append(el("span", "", `${f.feature}: ${f.value}`), el("span", "", direction));
      const bar = el("div", "bar");
      bar.setAttribute("aria-hidden", "true");
      const down = el("div"), up = el("div");
      const mark = el("i", f.effect > 0 ? "up" : "down");
      mark.style.width = `${Math.abs(f.effect) / max * 100}%`;
      (f.effect > 0 ? up : down).append(mark);
      bar.append(down, up);
      row.append(bar);
      panel.append(row);
    }
  }
  result.replaceChildren(panel);
}

async function update() {
  markSelected();
  const features = values();
  const request = ++latestRequest;
  if (Object.values(features).every(v => v === "?")) return renderEmpty();
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    if (!response.ok) throw new Error(response.status);
    const data = await response.json();
    if (request === latestRequest) renderResult(data);
  } catch {
    if (request === latestRequest) renderError();
  }
}

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderMetrics(m) {
  const table = document.getElementById("metrics");
  const head = table.createTHead().insertRow();
  for (const title of ["Model", "New rows, known species", "Unseen species", "Unseen poisonous caught"]) {
    head.append(el("th", "", title));
  }
  const body = table.createTBody();
  for (const [name, row] of Object.entries(m.model_comparison)) {
    const tr = body.insertRow();
    if (name === m.model) tr.className = "chosen";
    for (const cell of [name === m.model ? `${name} (used here)` : name, pct(row.random_split_accuracy),
                        pct(row.unseen_species_accuracy), pct(row.unseen_species_poison_recall)]) {
      tr.insertCell().textContent = cell;
    }
  }
  const unseen = m.all_unseen_species["0.0"];
  const known = m.dev_known_species["0.0"];
  document.getElementById("metrics-note").textContent =
    `Accuracy of the models alone, each tested on species left out of training. The full system adds the odor rule and the safety rule. ` +
    `With each of the ${m.species} species held out in turn, it called ${pct(unseen.false_edible_rate)} of poisonous mushrooms edible and flagged ` +
    `${pct(unseen.poison_flagged_poisonous)} as poisonous, while ${pct(known.edible_coverage)} of edible mushrooms from known species got an edible answer. ` +
    `The odor rule was added after testing began, so treat these as estimates. They can't rule out mistakes on other species.`;
}

async function loadExample(kind) {
  const response = await fetch(`/api/example/${kind}`);
  const example = await response.json();
  for (const s of selects()) s.value = example[s.name] ?? "?";
  update();
}

async function init() {
  const [schema, metrics] = await Promise.all(
    ["/api/schema", "/api/metrics"].map(url => fetch(url).then(r => r.json())));
  buildForm(schema);
  renderMetrics(metrics);
  renderEmpty();
  form.addEventListener("change", update);
  document.querySelectorAll("[data-example]").forEach(button =>
    button.addEventListener("click", () => loadExample(button.dataset.example)));
  document.getElementById("reset").addEventListener("click", () => {
    form.reset();
    update();
  });
}

init().catch(renderError);
