import { formatNumber, formatScalar } from "./format.js";
import { asNumber, asRecord, asString, countEntries, numericRecord, scalarRecord } from "./json.js";

export interface CertificateModel {
  name: string | null;
  sha256: string | null;
  rows: number | null;
  cols: number | null;
  nnz: number | null;
  sense: string | null;
  source: string | null;
}

export interface CertificateView {
  schema: string | null;
  createdUtc: string | null;
  verdict: string | null;
  status: string | null;
  objective: number | null;
  dualObjective: number | null;
  tolerance: number | null;
  residuals: Record<string, number | null>;
  evidenceKind: string | null;
  evidenceCheck: Record<string, string | number | boolean | null>;
  vectorCount: number | null;
  engine: Record<string, string | number | boolean | null>;
  environment: Record<string, string | number | boolean | null>;
  model: CertificateModel;
  solutionColumns: number | null;
  solutionRows: number | null;
}

export interface VerifyCheck {
  name: string;
  ok: boolean;
  detail: string;
}

export interface VerifyView {
  passed: boolean | null;
  text: string | null;
  model: string | null;
  rows: number | null;
  cols: number | null;
  verdict: string | null;
  tolerance: number | null;
  exact: boolean | null;
  checks: VerifyCheck[];
}

function modelOf(value: unknown): CertificateModel {
  const record = asRecord(value);
  return {
    name: asString(record?.name),
    sha256: asString(record?.sha256),
    rows: asNumber(record?.rows),
    cols: asNumber(record?.cols),
    nnz: asNumber(record?.nnz),
    sense: asString(record?.sense),
    source: asString(record?.source),
  };
}

export function parseCertificate(raw: unknown): CertificateView {
  const value = typeof raw === "string" ? JSON.parse(raw) as unknown : raw;
  const record = asRecord(value);
  if (!record) throw new TypeError("certificate is not a JSON object");
  const evidence = asRecord(record.evidence);
  const solution = asRecord(record.solution);
  return {
    schema: asString(record.schema),
    createdUtc: asString(record.created_utc),
    verdict: asString(record.verdict),
    status: asString(record.status),
    objective: asNumber(record.objective),
    dualObjective: asNumber(record.dual_objective),
    tolerance: asNumber(record.tolerance),
    residuals: numericRecord(record.residuals),
    evidenceKind: asString(evidence?.kind),
    evidenceCheck: scalarRecord(evidence?.check),
    vectorCount: countEntries(evidence?.vector),
    engine: scalarRecord(record.engine),
    environment: scalarRecord(record.environment),
    model: modelOf(record.model),
    solutionColumns: countEntries(solution?.x),
    solutionRows: countEntries(solution?.y),
  };
}

export function certificateSummary(view: CertificateView): [string, string][] {
  const rows: [string, string][] = [
    ["schema", view.schema ?? "—"],
    ["created_utc", view.createdUtc ?? "—"],
    ["verdict", view.verdict ?? "—"],
    ["status", view.status ?? "—"],
    ["objective", formatNumber(view.objective)],
    ["dual_objective", formatNumber(view.dualObjective)],
    ["tolerance", formatNumber(view.tolerance)],
    ["model.name", view.model.name ?? "—"],
    ["model.sha256", view.model.sha256 ?? "—"],
    ["model.rows", formatNumber(view.model.rows)],
    ["model.cols", formatNumber(view.model.cols)],
    ["model.nnz", formatNumber(view.model.nnz)],
    ["model.sense", view.model.sense ?? "—"],
    ["model.source", view.model.source ?? "—"],
    ["evidence.kind", view.evidenceKind ?? "—"],
    ["evidence.vector_count", formatNumber(view.vectorCount)],
    ["solution.columns", formatNumber(view.solutionColumns)],
    ["solution.rows", formatNumber(view.solutionRows)],
  ];
  for (const [key, value] of Object.entries(view.residuals)) rows.push([`residuals.${key}`, formatNumber(value)]);
  for (const [key, value] of Object.entries(view.evidenceCheck)) rows.push([`evidence.check.${key}`, formatScalar(value)]);
  for (const [key, value] of Object.entries(view.engine)) rows.push([`engine.${key}`, formatScalar(value)]);
  for (const [key, value] of Object.entries(view.environment)) rows.push([`environment.${key}`, formatScalar(value)]);
  return rows;
}

export function readVerifyReport(raw: unknown): VerifyView {
  const record = asRecord(raw);
  if (!record) throw new TypeError("verify response is not a JSON object");
  const checks: VerifyCheck[] = [];
  if (Array.isArray(record.checks)) {
    for (const row of record.checks) {
      if (!Array.isArray(row) || row.length < 2) continue;
      checks.push({
        name: asString(row[0]) ?? String(row[0]),
        ok: row[1] === true,
        detail: asString(row[2]) ?? (row[2] == null ? "" : String(row[2])),
      });
    }
  }
  return {
    passed: record.passed === true ? true : record.passed === false ? false : null,
    text: asString(record.text),
    model: asString(record.model),
    rows: asNumber(record.rows),
    cols: asNumber(record.cols),
    verdict: asString(record.verdict),
    tolerance: asNumber(record.tolerance),
    exact: record.exact === true ? true : record.exact === false ? false : null,
    checks,
  };
}
