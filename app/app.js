// Front-end for the FastAPI backend (same origin). Plain JS, no build step. Every server string is inserted with
// textContent (never innerHTML), so evidence snippets cannot inject markup.
"use strict";

// Gallery inputs live in examples.json (also read by tests/test_api.py::test_gallery_examples_offline).
const ICONS = { "REAL": "✔", "FAKE": "✖", "PARTIALLY TRUE": "◐", "MISLEADING": "⚠", "UNVERIFIABLE": "?" };
const STANCE = { SUPPORTS: "Supports", REFUTES: "Contradicts", NEUTRAL: "Neutral" };
const $ = (id) => document.getElementById(id);
let inputType = "headline", last = null, serverMode = "online";

function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null) continue;
    if (k === "class") n.className = v; else if (k in n && k !== "list") n[k] = v; else n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) n.append(k instanceof Node ? k : String(k));
  return n;
}
const fmt = (x, d = 2) => (x == null ? "n/a" : Number(x).toFixed(d));
const band = (c) => (c < 0.5 ? "Low" : c <= 0.75 ? "Medium" : "High");

// ---------------------------------------------------------------- routing + theme
function route() {
  const about = location.hash === "#about";
  $("screen-about").hidden = !about;
  $("screen-analyse").hidden = about;
  document.querySelectorAll("[data-nav]").forEach((a) =>
    ((a.dataset.nav === "about") === about ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
}
addEventListener("hashchange", route);
$("theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* storage blocked: theme lasts this page */ }
});

// ---------------------------------------------------------------- tabs (WAI-ARIA tabs pattern, arrow keys)
const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(t, focus) {
  tabs.forEach((x) => {
    const on = x === t;
    x.setAttribute("aria-selected", on); x.tabIndex = on ? 0 : -1;
    $(x.getAttribute("aria-controls")).hidden = !on;
  });
  inputType = t.dataset.type;
  if (focus) t.focus();
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => selectTab(t));
  t.addEventListener("keydown", (e) => {
    const j = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (j == null) return;
    e.preventDefault(); selectTab(tabs[(j + tabs.length) % tabs.length], true);
  });
});

// ---------------------------------------------------------------- config (mode, sources) for the form + About
fetch("/config").then((r) => r.json()).then((c) => {
  serverMode = c.mode;
  if (c.mode === "offline") {
    document.querySelector('input[value="offline"]').checked = true;
    document.querySelector('input[value="online"]').disabled = true;
    $("mode-note").textContent = "This server runs in offline mode (no network); online is unavailable.";
  }
  $("about-sources").replaceChildren(...c.sources.map((s) => el("li", {}, s)));
  $("about-thresholds").textContent = `Thresholds: stance agreement tau = ${c.thresholds.tau}, old-news gap N = ${c.thresholds.N_months} months, top-k = ${c.thresholds.k}. Server mode: ${c.mode}.`;
}).catch(() => { $("mode-note").textContent = "Could not reach /config."; });

// ---------------------------------------------------------------- examples
fetch("examples.json").then((r) => r.json()).then((list) => $("examples").replaceChildren(...list.map((ex) => el("li", {},
  el("button", { type: "button", onclick: () => runExample(ex) },
    el("span", { class: "ex-tag" }, `${ICONS[ex.expect]} ${ex.expect} (${ex.input_type})`),
    el("span", {}, ex.title || ex.text))))));

function runExample(ex) {
  selectTab(tabs.find((t) => t.dataset.type === ex.input_type));
  if (ex.input_type === "headline") $("in-headline").value = ex.text;
  else { $("in-title").value = ex.title || ""; $("in-article").value = ex.text; }
  document.querySelector('input[value="offline"]').checked = true;
  submit();
}

// ---------------------------------------------------------------- submit
function payload() {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const p = { input_type: inputType, mode, published: $("in-date").value || null };
  if (inputType === "headline") p.text = $("in-headline").value.trim();
  else if (inputType === "article") { p.title = $("in-title").value.trim() || null; p.text = $("in-article").value.trim(); }
  else p.url = $("in-url").value.trim();
  return p;
}
function invalid(p) {
  if (p.input_type === "url") {
    if (!/^https?:\/\/\S+$/i.test(p.url || "")) return "Enter a full URL starting with http:// or https://.";
    if (p.mode === "offline") return "URL input needs online mode; paste the article text instead.";
  } else if (!p.text && !p.title) return "Enter some text to check.";
  return null;
}
$("form").addEventListener("submit", (e) => { e.preventDefault(); submit(); });
async function submit() {
  const p = payload(), err = invalid(p);
  if (err) { $("status").textContent = err; return; }
  $("go").disabled = true; $("status").textContent = "Analysing... (the first request loads the models, ~10 s)";
  try {
    const r = await fetch("/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) });
    const body = await r.json();
    if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
    last = body;
    render(body, p);
    $("status").textContent = `Done in ${(body.latency_ms / 1000).toFixed(1)} s.`;
    if (p.input_type !== "url") loadLime([p.title, p.text].filter(Boolean).join("\n"));
    else $("lime").replaceChildren(el("p", { class: "none" }, "Token highlights are computed for pasted headline / article text only."));
  } catch (e) {
    $("status").textContent = `Error: ${e.message}`;
  } finally { $("go").disabled = false; }
}

// ---------------------------------------------------------------- render
function render(r, p) {
  $("result").hidden = false;
  const v = $("verdict");
  v.dataset.v = r.verdict;
  $("v-icon").textContent = ICONS[r.verdict] || "";
  $("v-text").textContent = r.verdict;
  $("v-meter").value = r.confidence;
  $("v-conf").textContent = `${fmt(r.confidence)} (${band(r.confidence)})`;
  $("v-explanation").textContent = r.explanation || r.reason;
  $("v-rule").textContent = `Rule ${r.rule_fired}`;
  $("v-reason").textContent = r.reason;
  $("v-uncertainty").textContent = r.uncertainty;
  renderClaims(r.claims || []);
  renderDates(r.temporal || {});
  renderEvidence(r.claims || []);
  const c = r.classifier || {};
  $("clf").textContent = c.model ? `Model ${c.model}: P(fake-style) = ${fmt(c.p_fake)} (${c.label}). ` : "No style classifier ran (non-English input or no checkpoint).";
  $("lime").replaceChildren(el("p", { class: "muted" }, "Computing token highlights..."));
  renderDetails(r);
  v.focus();
}

function renderClaims(claims) {
  if (!claims.length) { $("claims").replaceChildren(el("li", { class: "none" }, "No claims were checked (see the reason above).")); return; }
  $("claims").replaceChildren(...claims.map((c) => {
    const main = (c.evidence || []).reduce((a, e) => (a[e.stance] = (a[e.stance] || 0) + 1, a), {});
    const top = main.REFUTES > (main.SUPPORTS || 0) ? "REFUTES" : main.SUPPORTS ? "SUPPORTS" : "NEUTRAL";
    return el("li", {},
      el("q", {}, c.text),
      el("span", { class: `chip s-${top}` }, c.evidence && c.evidence.length ? STANCE[top] : "No evidence"),
      el("div", { class: "small muted" }, `${c.is_title ? "Headline · " : ""}check-worthiness ${fmt(c.score)} · ${c.stance_summary || ""}`));
  }));
}

function renderDates(t) {
  const rows = [
    ["Input date", t.input_date ? `${t.input_date}${t.input_date && !$("in-date").value ? " (today; no date given)" : ""}` : "n/a"],
    ["Event date (supporting evidence)", t.event_date || "unknown (no dated supporting evidence)"],
    ["Gap", t.gap_months != null ? `${t.gap_months} months` : "n/a"],
    ["Evidence dates", (t.evidence_dates || []).join(", ") || "none"],
    ["Recency words (\"breaking\", \"today\"...)", t.recency_cue ? "yes" : "no"],
    ["Old news presented as new", t.old_news ? "YES" : "no"],
    ["Place / year out of context", t.out_of_context ? "YES" : "no"],
    ["Headline contradicted by body", t.mismatch_headline ? `YES (p = ${fmt(t.mismatch_p)})` : "no"],
  ];
  if (!Object.keys(t).length) rows.splice(0, rows.length, ["Date check", "not run (no claims to check)"]);
  $("dates").replaceChildren(...rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)]));
  const w = [];
  if (t.old_news) w.push(`Old news presented as new: supporting reports are dated ${t.event_date}, the input is dated ${t.input_date} - a gap of ${t.gap_months} months.`);
  if (t.out_of_context) w.push("The places or years in the claim do not match the supporting reports.");
  if (t.mismatch_headline) w.push("The article body contradicts its own headline.");
  $("date-warn").hidden = !w.length;
  $("date-warn").textContent = w.map((x) => "⚠ " + x).join(" ");
}

function evCard(e) {
  const src = e.source_label || e.source || "unknown source";
  const date = e.date || "date unknown";
  return el("article", { class: "ev-card" },
    el("p", {}, el("strong", {}, src), " ", el("span", { class: "badge" }, `Tier ${e.tier ?? "?"}`), " ",
      el("span", { class: `chip s-${e.stance}` }, STANCE[e.stance] || e.stance)),
    el("p", { class: "ev-meta" }, `${date} · ${STANCE[e.stance] || e.stance} (p ${fmt(e.p)}) · similarity ${fmt(e.cosine)}` +
      (e.rating ? ` · fact-check rating: "${e.rating}"` : "") + ` · ${e.origin === "offline_index" ? "offline index" : "online"}`),
    e.title ? el("p", {}, el("em", {}, e.title)) : null,
    el("p", {}, el("q", {}, e.snippet || "")),
    e.nli ? el("p", { class: "ev-meta" }, `NLI: supports ${fmt(e.nli.SUPPORTS)} / refutes ${fmt(e.nli.REFUTES)} / neutral ${fmt(e.nli.NEUTRAL)}` + (e.record_id ? ` · record ${e.record_id}` : "")) : null,
    e.url ? el("p", {}, el("a", { href: e.url, target: "_blank", rel: "noopener noreferrer" }, `Open ${src} (${date})`)) : null);
}

function renderEvidence(claims) {
  const box = $("evidence");
  if (!claims.length) { box.replaceChildren(el("p", { class: "none card" }, "No evidence found (no claims were checked).")); return; }
  box.replaceChildren(...claims.map((c, i) => {
    const ev = c.evidence || [];
    const col = (title, st) => {
      const items = ev.filter((e) => e.stance === st);
      return el("section", {}, el("h4", {}, `${title} (${items.length})`),
        items.length ? items.map(evCard) : el("p", { class: "none" }, "None found."));
    };
    return el("div", { class: "card" },
      el("p", { class: "ev-claim" }, `Claim ${i + 1}: `, el("q", {}, c.text)),
      ev.length ? el("div", { class: "ev-cols" }, col("Supporting", "SUPPORTS"), col("Contradicting", "REFUTES"), col("Neutral / related", "NEUTRAL"))
        : el("p", { class: "none" }, "No evidence found for this claim."));
  }));
}

function renderDetails(r) {
  const rows = [["Request id", r.request_id], ["Mode", r.mode], ["Rule fired", r.rule_fired], ["Evidence items", r.evidence_count],
    ["Latency", `${r.latency_ms} ms`], ["Signals", el("pre", {}, JSON.stringify(r.signals, null, 1))]];
  if (r.timings_ms) rows.push(["Timings (ms)", el("pre", {}, JSON.stringify(r.timings_ms))]);
  $("details").replaceChildren(...rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)]));
}

// ---------------------------------------------------------------- LIME highlights (async, after the verdict is shown)
async function loadLime(text) {
  try {
    const r = await fetch("/explain", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    const x = await r.json();
    if (!r.ok) throw new Error(x.detail || r.status);
    if (last) last.lime = x;
    const w = new Map(x.tokens.map(([t, v]) => [t, v]));
    const marked = text.split(/(\W+)/).map((part) => {
      if (!w.has(part)) return part;
      const v = w.get(part);
      return el("mark", { class: v > 0 ? "fake" : "real", title: `${v > 0 ? "+" : ""}${v.toFixed(3)} towards ${v > 0 ? "fake" : "real"}-style` }, part);
    });
    $("lime").replaceChildren(
      el("p", { class: "small muted" }, `Solid underline = pushes towards fake-style; dashed = towards real-style. ${x.model}, ${x.num_samples} LIME samples, ${x.cached ? "cached" : x.ms + " ms"}.`),
      el("p", {}, marked),
      el("ol", { class: "small" }, x.tokens.map(([t, v]) => el("li", {}, `${t}: ${v > 0 ? "+" : ""}${v.toFixed(3)} (${v > 0 ? "fake" : "real"}-style)`))));
  } catch (e) {
    $("lime").replaceChildren(el("p", { class: "none" }, `Token highlights unavailable (${e.message}).`));
  }
}

// ---------------------------------------------------------------- download + reset
$("download").addEventListener("click", () => {
  if (!last) return;
  const a = el("a", { href: URL.createObjectURL(new Blob([JSON.stringify(last, null, 2)], { type: "application/json" })), download: `analysis-${last.request_id}.json` });
  document.body.append(a); a.click(); a.remove();
});
$("again").addEventListener("click", () => { $("result").hidden = true; tabs.find((t) => t.dataset.type === inputType).focus(); scrollTo(0, 0); });

route();
