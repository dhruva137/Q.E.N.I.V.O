import { parseAuditLog, type AuditRecord } from "../lib/audit.js";
import { certificateSummary, parseCertificate, readVerifyReport } from "../lib/certificate.js";
import { comparePlans, type MarginalChange } from "../lib/compare.js";
import { presentExplanation, type ExplanationView } from "../lib/explain.js";
import { formatNumber, formatScalar } from "../lib/format.js";
import { readInfo, type InfoView } from "../lib/info.js";
import { asRecord, asString } from "../lib/json.js";
import { library } from "../lib/manifest.js";
import { guessFormat } from "../lib/modeltext.js";
import { samples } from "../lib/sample.js";
import { readSolve, type SolveView } from "../lib/solve.js";
import { request, setAuth } from "./api.js";
import { byId, clear, defs, el, table, type Cell } from "./dom.js";

interface Slot {
  view: SolveView | null;
  error: string | null;
}

const slots: Record<"A" | "B", Slot> = {
  A: { view: null, error: null },
  B: { view: null, error: null },
};

let modelText = "";
let certificateRaw: unknown = null;
let auditRecords: AuditRecord[] = [];
let auditSkipped = 0;
let auditFiles: string[] = [];
let selected = library[0]?.name ?? "";

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function setNotice(message: string): void {
  const notice = byId("notice");
  notice.hidden = message.length === 0;
  notice.textContent = message;
}

function syncAuth(): void {
  setAuth({
    token: byId<HTMLInputElement>("token").value.trim(),
    user: byId<HTMLInputElement>("user").value.trim(),
  });
}

async function run(button: HTMLButtonElement, fn: () => Promise<void>): Promise<void> {
  button.disabled = true;
  try {
    syncAuth();
    await fn();
  } catch (err) {
    setNotice(messageOf(err));
  } finally {
    button.disabled = false;
  }
}

function showView(name: string): void {
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-view]")) {
    const on = button.dataset.view === name;
    if (on) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
    const section = document.getElementById(`view-${button.dataset.view ?? ""}`);
    if (section) section.hidden = !on;
  }
}

function currentModel(): string {
  const box = byId<HTMLTextAreaElement>("model-text");
  if (!box.hidden) modelText = box.value;
  if (!modelText.trim()) throw new Error("Load a model in the workspace first.");
  return modelText;
}

function modelBody(model: string): Record<string, unknown> {
  const format = byId<HTMLSelectElement>("format").value;
  const body: Record<string, unknown> = { model };
  if (format) body.format = format;
  return body;
}

function solveBody(model: string): Record<string, unknown> {
  const engine = byId<HTMLInputElement>("engine").value.trim() || "auto";
  const tolRaw = byId<HTMLInputElement>("tol").value.trim();
  const backend = byId<HTMLSelectElement>("backend").value;
  const body: Record<string, unknown> = { ...modelBody(model), engine, backend };
  if (tolRaw) {
    const tol = Number(tolRaw);
    if (!Number.isFinite(tol)) throw new Error("Tolerance must be a number.");
    body.tol = tol;
  }
  return body;
}

function loadModel(label: string, text: string): void {
  modelText = text;
  const box = byId<HTMLTextAreaElement>("model-text");
  const show = text.length > 0 && text.length < 100_000;
  box.hidden = !show;
  box.value = show ? text : "";
  const lines = text ? text.split("\n").length : 0;
  const guess = text ? guessFormat(text) : "mps";
  byId("model-meta").textContent = text
    ? `${label} · ${text.length} bytes · ${lines} lines · server format guess ${guess}`
    : "No model loaded.";
}

async function useFile(file: File): Promise<void> {
  loadModel(file.name, await file.text());
}

function renderInfo(view: InfoView | null, error: string | null): void {
  const host = byId("server-info");
  clear(host);
  if (error) {
    host.append(el("p", { className: "warn", text: error }));
    return;
  }
  if (!view) return;
  const pairs: [string, string][] = [
    ["version", view.version ?? "—"],
    ["provenance.clean", view.clean == null ? "—" : String(view.clean)],
    ["forbidden_packages", view.forbiddenPackages.join(", ") || "none"],
    ["tripwire_hits", view.tripwireHits == null ? "—" : String(view.tripwireHits)],
    ["tripwires_installed", view.tripwiresInstalled == null ? "—" : String(view.tripwiresInstalled)],
    ["scipy_optimize_imported", view.scipyOptimizeImported == null ? "—" : String(view.scipyOptimizeImported)],
  ];
  for (const [key, value] of view.environment) pairs.push([`environment.${key}`, value]);
  host.append(defs(pairs));
}

async function refreshInfo(): Promise<void> {
  const statusEl = byId("api-status");
  syncAuth();
  let upstream = "";
  try {
    const raw = await request("/console/status");
    upstream = asString(asRecord(raw)?.upstream) ?? "";
  } catch (err) {
    statusEl.textContent = `Console host unavailable. ${messageOf(err)}`;
    statusEl.className = "status bad";
    renderInfo(null, "GET /console/status failed.");
    return;
  }
  try {
    const info = readInfo(await request("/api/info"));
    const mark = info.clean ? "clean" : "VIOLATION";
    statusEl.textContent = `Proxy ${upstream} · API reachable · qenivo ${info.version ?? "?"} · provenance ${mark}`;
    statusEl.className = info.clean ? "status ok" : "status bad";
    renderInfo(info, null);
  } catch (err) {
    statusEl.textContent = `Proxy ${upstream} · solver API unreachable. Start: qenivo serve --host 127.0.0.1 --port 8765`;
    statusEl.className = "status bad";
    renderInfo(null, messageOf(err));
  }
}

function renderLibrary(): void {
  const body = byId<HTMLTableSectionElement>("library-body");
  clear(body);
  for (const model of library) {
    const row = document.createElement("tr");
    if (model.name === selected) row.className = "selected";
    row.tabIndex = 0;
    const cells = [model.name, model.problemClass, model.sector, model.summary];
    for (const text of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      row.append(td);
    }
    const open = () => {
      selected = model.name;
      renderLibrary();
    };
    row.addEventListener("click", open);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });
    body.append(row);
  }
  const detail = byId("library-detail");
  clear(detail);
  const model = library.find((item) => item.name === selected);
  if (!model) {
    detail.append(el("p", { text: "The local manifest has no models." }));
    return;
  }
  const params = model.parameters.map((item) => `${item.name}=${item.default}`).join(", ");
  detail.append(
    el("h3", { text: model.name }),
    defs([
      ["sector", model.sector],
      ["class", model.problemClass],
      ["formulation", model.formulation],
      ["parameters", params || "—"],
    ]),
    el("p", { text: model.detail }),
    el("pre", { className: "command", text: model.command }),
  );
  const copy = el("button", { className: "ghost", text: "Copy command" }) as HTMLButtonElement;
  copy.type = "button";
  copy.addEventListener("click", () => {
    void navigator.clipboard.writeText(model.command).then(
      () => setNotice("Copied the model command."),
      () => setNotice(model.command),
    );
  });
  detail.append(copy);
}

function visibleAudit(): AuditRecord[] {
  const query = byId<HTMLInputElement>("history-filter").value.trim().toLowerCase();
  const rows = [...auditRecords].sort((a, b) => (b.utc ?? "").localeCompare(a.utc ?? ""));
  if (!query) return rows;
  return rows.filter((record) =>
    [record.utc, record.event, record.path, record.user, record.auth, record.client, record.verdict,
      record.engine, record.model_sha256, record.error, record.reason, record.objective, record.status]
      .some((value) => value != null && String(value).toLowerCase().includes(query)));
}

function renderHistory(): void {
  const shown = visibleAudit();
  byId("history-meta").textContent =
    `${auditFiles.join(", ") || "no file"} · ${auditRecords.length} lines · ${auditSkipped} skipped · ${shown.length} shown`;
  const body = byId<HTMLTableSectionElement>("history-body");
  clear(body);
  if (shown.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 14;
    cell.textContent = auditRecords.length === 0
      ? "No audit lines loaded."
      : "No lines match the filter.";
    row.append(cell);
    body.append(row);
    return;
  }
  for (const record of shown) {
    const sha = record.model_sha256;
    const cells: Cell[] = [
      record.utc ?? "—",
      record.event ?? "—",
      record.path ?? "—",
      record.user ?? "—",
      record.auth ?? "—",
      record.client ?? "—",
      record.verdict ?? "—",
      { text: formatNumber(record.objective), numeric: true },
      record.engine ?? "—",
      { text: formatNumber(record.seconds), numeric: true },
      { text: record.status == null ? "—" : String(record.status), numeric: true },
      { text: sha ? sha.slice(0, 12) : "—", title: sha ?? undefined },
      record.error ?? "—",
      record.reason ?? "—",
    ];
    const row = document.createElement("tr");
    for (const cell of cells) {
      const td = document.createElement("td");
      if (typeof cell === "string") td.textContent = cell;
      else {
        td.textContent = cell.text;
        if (cell.numeric) td.className = "num";
        if (cell.title) td.title = cell.title;
      }
      row.append(td);
    }
    body.append(row);
  }
}

function pairTable(title: string, headers: [string, string], rows: [string, number][]): HTMLElement {
  const wrap = el("div", { className: "block" }, el("h3", { text: title }));
  if (rows.length === 0) {
    wrap.append(el("p", { className: "muted", text: "None returned." }));
    return wrap;
  }
  wrap.append(el("div", { className: "table-wrap" }, table(headers, rows.map(([name, value]) => [
    name,
    { text: formatNumber(value), numeric: true },
  ]))));
  return wrap;
}

function renderSlot(host: HTMLElement, which: "A" | "B"): void {
  clear(host);
  const slot = slots[which];
  host.append(el("h3", { text: `Plan ${which}` }));
  if (slot.error) host.append(el("p", { className: "warn", text: slot.error }));
  const view = slot.view;
  if (!view) {
    host.append(el("p", { className: "muted", text: "Not solved yet." }));
    return;
  }
  const pairs: [string, string][] = [
    ["kind", view.kind],
    ["model", view.model ?? "—"],
    ["size", view.size ?? "—"],
    ["verdict", view.verdict ?? "—"],
    ["status", view.status ?? "—"],
    ["feasible", view.feasible == null ? "—" : String(view.feasible)],
    ["objective", formatNumber(view.objective)],
    ["max_violation", formatNumber(view.maxViolation)],
    ["iterations", formatNumber(view.iterations)],
    ["time", formatNumber(view.time)],
    ["history", formatNumber(view.historyPoints)],
  ];
  for (const [key, value] of Object.entries(view.engine)) pairs.push([`engine.${key}`, formatScalar(value)]);
  for (const [key, value] of Object.entries(view.residuals)) pairs.push([`residuals.${key}`, formatNumber(value)]);
  if (view.explanationProof) pairs.push(["explanation.proof", view.explanationProof]);
  host.append(el("p", { className: "objective", text: formatNumber(view.objective) }));
  if (view.verdict) {
    const verdict = el("p", { className: `verdict ${view.verdict}`, text: view.verdict });
    host.append(verdict);
  }
  host.append(defs(pairs));
  host.append(pairTable("top_values", ["column", "value"], view.topValues));
  host.append(pairTable("marginals", ["row", "marginal"], view.marginals));
}

function renderDelta(changes: MarginalChange[]): void {
  const host = byId("compare-delta");
  clear(host);
  if (changes.length === 0) {
    host.append(el("p", { text: "No marginal value changed among the returned rows." }));
    return;
  }
  const max = Math.max(...changes.map((change) => Math.abs(change.delta)));
  const tableEl = document.createElement("table");
  const head = document.createElement("tr");
  for (const label of ["row", "plan A", "plan B", "B − A", ""]) {
    const th = document.createElement("th");
    th.textContent = label;
    head.append(th);
  }
  const thead = document.createElement("thead");
  thead.append(head);
  const tbody = document.createElement("tbody");
  for (const change of changes) {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = change.name;
    const left = document.createElement("td");
    left.className = "num";
    left.textContent = change.left == null ? "absent" : formatNumber(change.left);
    const right = document.createElement("td");
    right.className = "num";
    right.textContent = change.right == null ? "absent" : formatNumber(change.right);
    const delta = document.createElement("td");
    delta.className = "num";
    delta.textContent = formatNumber(change.delta);
    const barCell = document.createElement("td");
    const bar = document.createElement("span");
    bar.className = "bar";
    const fill = document.createElement("i");
    fill.style.width = `${(Math.abs(change.delta) / max) * 100}%`;
    if (change.delta < 0) fill.className = "neg";
    bar.append(fill);
    barCell.append(bar);
    row.append(name, left, right, delta, barCell);
    tbody.append(row);
  }
  tableEl.append(thead, tbody);
  host.append(el("div", { className: "table-wrap" }, tableEl));
}

function renderCompare(): void {
  renderSlot(byId("plan-a"), "A");
  renderSlot(byId("plan-b"), "B");
  const summary = byId("compare-summary");
  clear(summary);
  const left = slots.A.view;
  const right = slots.B.view;
  if (!left || !right) {
    summary.append(el("p", { text: "Solve plan A and plan B to rank marginal-value changes." }));
    clear(byId("compare-delta"));
    return;
  }
  const compared = comparePlans(
    { objective: left.objective, marginals: left.marginals },
    { objective: right.objective, marginals: right.marginals },
    20,
  );
  summary.append(defs([
    ["plan A objective", formatNumber(compared.objectiveLeft)],
    ["plan B objective", formatNumber(compared.objectiveRight)],
    ["B − A", formatNumber(compared.objectiveChange)],
  ]));
  renderDelta(compared.changes);
}

function renderExplain(view: ExplanationView): void {
  const host = byId("explain-out");
  clear(host);
  const feasible = view.feasible == null ? "unknown" : view.feasible ? "feasible" : "infeasible";
  const tone = view.feasible === true ? "optimal" : view.feasible === false ? "infeasible" : "not_proven";
  host.append(el("p", { className: `verdict ${tone}`, text: feasible }));
  const pairs: [string, string][] = [
    ["kind", view.kind ?? "—"],
    ["total_relaxation", formatNumber(view.totalRelaxation)],
    ["proof", view.proof ?? "—"],
    ["conflict_rows", view.conflictRows.join(", ") || "—"],
  ];
  if (view.farkas) {
    pairs.push(
      ["farkas_check.valid", view.farkas.valid == null ? "—" : String(view.farkas.valid)],
      ["farkas_check.value", formatNumber(view.farkas.value)],
      ["farkas_check.violation", formatNumber(view.farkas.violation)],
      ["farkas_check.ratio", formatNumber(view.farkas.ratio)],
      ["farkas_check.reason", view.farkas.reason ?? "—"],
    );
  }
  host.append(defs(pairs));
  if (view.text) host.append(el("pre", { className: "report", text: view.text }));
  if (view.items.length === 0) {
    host.append(el("p", { className: "muted", text: "No conflicting limits in items." }));
    return;
  }
  host.append(el("div", { className: "table-wrap" }, table(
    ["constraint", "limit", "limit_value", "best_achievable", "shortfall", "issue"],
    view.items.map((item) => [
      item.constraint,
      item.limit ?? "—",
      { text: formatNumber(item.limitValue), numeric: true },
      { text: formatNumber(item.bestAchievable), numeric: true },
      { text: formatNumber(item.shortfall), numeric: true },
      item.issue,
    ]),
  )));
}

function renderCertificate(): void {
  const summary = byId("cert-summary");
  const pre = byId<HTMLPreElement>("cert-json");
  clear(summary);
  if (certificateRaw == null) {
    summary.append(el("p", { text: "No certificate loaded. Solve a plan, or open a certificate JSON file." }));
    pre.textContent = "";
    return;
  }
  let parsed: unknown = certificateRaw;
  if (typeof certificateRaw === "string") {
    parsed = JSON.parse(certificateRaw) as unknown;
    certificateRaw = parsed;
  }
  pre.textContent = JSON.stringify(parsed, null, 2);
  try {
    summary.append(defs(certificateSummary(parseCertificate(parsed))));
  } catch (err) {
    summary.append(el("p", { className: "warn", text: messageOf(err) }));
  }
}

function renderVerify(raw: unknown): void {
  const view = readVerifyReport(raw);
  const host = byId("verify-out");
  clear(host);
  const mark = view.passed == null ? "unknown" : view.passed ? "passed" : "rejected";
  const tone = view.passed === true ? "optimal" : "not_proven";
  host.append(el("p", { className: `verdict ${tone}`, text: mark }));
  host.append(defs([
    ["model", view.model ?? "—"],
    ["rows", formatNumber(view.rows)],
    ["cols", formatNumber(view.cols)],
    ["verdict", view.verdict ?? "—"],
    ["tolerance", formatNumber(view.tolerance)],
    ["exact", view.exact == null ? "—" : String(view.exact)],
  ]));
  if (view.text) host.append(el("pre", { className: "report", text: view.text }));
  for (const check of view.checks) {
    host.append(el("div", { className: check.ok ? "check pass" : "check fail" },
      el("strong", { text: check.ok ? "pass" : "fail" }),
      el("span", { text: check.name }),
      el("span", { className: "muted", text: check.detail })));
  }
}

function useSlotCertificate(which: "A" | "B"): void {
  const certificate = slots[which].view?.certificate;
  if (certificate == null) throw new Error(`Plan ${which} has no certificate.`);
  certificateRaw = certificate;
  showView("certificate");
  renderCertificate();
  setNotice(`Showing the certificate from plan ${which}.`);
}

function boot(): void {
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-view]")) {
    button.addEventListener("click", () => showView(button.dataset.view ?? "library"));
  }
  byId<HTMLInputElement>("token").addEventListener("change", () => { void refreshInfo(); });
  byId<HTMLInputElement>("user").addEventListener("change", () => { void refreshInfo(); });
  byId<HTMLInputElement>("model-file").addEventListener("change", (event) => {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    void useFile(file).catch((err: unknown) => setNotice(messageOf(err)));
    input.value = "";
  });
  byId("sample-optimal").addEventListener("click", () => loadModel(samples.diesel_week.label, samples.diesel_week.text));
  byId("sample-infeasible").addEventListener("click", () => loadModel(samples.bound_conflict.label, samples.bound_conflict.text));
  const workspace = byId("workspace");
  workspace.addEventListener("dragover", (event) => {
    event.preventDefault();
    workspace.classList.add("over");
  });
  workspace.addEventListener("dragleave", () => workspace.classList.remove("over"));
  workspace.addEventListener("drop", (event) => {
    event.preventDefault();
    workspace.classList.remove("over");
    const file = (event as DragEvent).dataTransfer?.files[0];
    if (file) void useFile(file).catch((err: unknown) => setNotice(messageOf(err)));
  });
  byId<HTMLInputElement>("audit-file").addEventListener("change", (event) => {
    const input = event.currentTarget as HTMLInputElement;
    const files = [...(input.files ?? [])];
    void (async () => {
      const chunks: string[] = [];
      for (const file of files) chunks.push(await file.text());
      const parsed = parseAuditLog(chunks.join("\n"));
      auditRecords = parsed.records;
      auditSkipped = parsed.skipped;
      auditFiles = files.map((file) => file.name);
      renderHistory();
      setNotice(`Loaded ${parsed.records.length} audit lines.`);
    })().catch((err: unknown) => setNotice(messageOf(err)));
    input.value = "";
  });
  byId<HTMLInputElement>("history-filter").addEventListener("input", () => renderHistory());
  byId<HTMLButtonElement>("solve-a").addEventListener("click", (event) => {
    void run(event.currentTarget as HTMLButtonElement, async () => {
      const raw = await request("/api/solve", solveBody(currentModel()));
      slots.A = { view: readSolve(raw), error: null };
      if (slots.A.view?.certificate != null) certificateRaw = slots.A.view.certificate;
      renderCompare();
      setNotice(`Plan A solved · ${slots.A.view?.verdict ?? slots.A.view?.status ?? slots.A.view?.kind} · objective ${formatNumber(slots.A.view?.objective)}`);
    });
  });
  byId<HTMLButtonElement>("solve-b").addEventListener("click", (event) => {
    void run(event.currentTarget as HTMLButtonElement, async () => {
      const raw = await request("/api/solve", solveBody(currentModel()));
      slots.B = { view: readSolve(raw), error: null };
      if (slots.B.view?.certificate != null) certificateRaw = slots.B.view.certificate;
      renderCompare();
      setNotice(`Plan B solved · ${slots.B.view?.verdict ?? slots.B.view?.status ?? slots.B.view?.kind} · objective ${formatNumber(slots.B.view?.objective)}`);
    });
  });
  byId<HTMLButtonElement>("use-cert-a").addEventListener("click", (event) => {
    void run(event.currentTarget as HTMLButtonElement, async () => useSlotCertificate("A"));
  });
  byId<HTMLButtonElement>("use-cert-b").addEventListener("click", (event) => {
    void run(event.currentTarget as HTMLButtonElement, async () => useSlotCertificate("B"));
  });
  byId<HTMLButtonElement>("go-value").addEventListener("click", (event) => {
    void run(event.currentTarget as HTMLButtonElement, async () => {
      const tLo = Number(byId<HTMLInputElement>("value-tlo").value);
      const tHi = Number(byId<HTMLInputElement>("value-thi").value);
      if (!Number.isFinite(tLo) || !Number.isFinite(tHi)) throw new Error("t_lo and t_hi must be numbers.");
      const raw = await request("/api/crude-value", {
        ...modelBody(currentModel()),
        column: byId<HTMLInputElement>("value-col").value.trim() || "crude_light",
        t_lo: tLo,
        t_hi: tHi,
      });
      const host = byId("value-out");
      clear(host);
      const rec = asRecord(raw) ?? {};
      const status = asString(rec.status) ?? "";
      const note = asString(rec.note) ?? "";
      if (status === "needs_w02_merge" || note === "needs w02 merge") {
        host.append(
          el("p", { className: "warn", text: "needs w02 merge" }),
          el("p", { text: asString(rec.detail) ?? "Merge branch w02-crude-value for the live curve." }),
          el("pre", { className: "report", text: JSON.stringify(raw, null, 2) }),
        );
        setNotice("Crude valuation stub: needs w02 merge.");
        return;
      }
      host.append(
        el("p", { className: "verdict optimal", text: status || "ok" }),
        el("pre", { className: "report", text: JSON.stringify(raw, null, 2) }),
      );
      setNotice("Crude valuation response loaded.");
    });
  });
  byId<HTMLButtonElement>("go-explain").addEventListener("click", (event) => {
    void run(event.currentTarget as HTMLButtonElement, async () => {
      const raw = await request("/api/explain", modelBody(currentModel()));
      renderExplain(presentExplanation(raw));
      setNotice("Explanation loaded.");
    });
  });
  byId<HTMLInputElement>("cert-file").addEventListener("change", (event) => {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    void file.text().then((text) => {
      certificateRaw = JSON.parse(text) as unknown;
      renderCertificate();
      showView("certificate");
      setNotice(`Opened ${file.name}.`);
    }).catch((err: unknown) => setNotice(messageOf(err)));
    input.value = "";
  });
  byId<HTMLButtonElement>("go-verify").addEventListener("click", (event) => {
    void run(event.currentTarget as HTMLButtonElement, async () => {
      if (certificateRaw == null) throw new Error("Load a certificate first.");
      const raw = await request("/api/verify", {
        ...modelBody(currentModel()),
        certificate: certificateRaw,
        exact: byId<HTMLInputElement>("verify-exact").checked,
      });
      renderVerify(raw);
      setNotice("Verification finished.");
    });
  });
  renderLibrary();
  renderHistory();
  renderCompare();
  renderCertificate();
  void refreshInfo();
}

boot();
