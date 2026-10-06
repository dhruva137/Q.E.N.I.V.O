import { asBoolean, asNumber, asRecord, asString } from "./json.js";

export interface ExplainItem {
  constraint: string;
  issue: string;
  limit: string | null;
  limitValue: number | null;
  bestAchievable: number | null;
  shortfall: number | null;
}

export interface FarkasView {
  valid: boolean | null;
  value: number | null;
  violation: number | null;
  ratio: number | null;
  reason: string | null;
}

export interface ExplanationView {
  feasible: boolean | null;
  kind: string | null;
  proof: string | null;
  text: string | null;
  totalRelaxation: number | null;
  conflictRows: string[];
  farkas: FarkasView | null;
  items: ExplainItem[];
}

function itemOf(value: unknown): ExplainItem | null {
  const record = asRecord(value);
  if (!record) return null;
  return {
    constraint: asString(record.constraint) ?? "—",
    issue: asString(record.issue) ?? "",
    limit: asString(record.limit),
    limitValue: asNumber(record.limit_value),
    bestAchievable: asNumber(record.best_achievable),
    shortfall: asNumber(record.shortfall),
  };
}

function farkasOf(value: unknown): FarkasView | null {
  const record = asRecord(value);
  if (!record) return null;
  return {
    valid: asBoolean(record.valid),
    value: asNumber(record.value),
    violation: asNumber(record.violation),
    ratio: asNumber(record.ratio),
    reason: asString(record.reason),
  };
}

export function presentExplanation(raw: unknown): ExplanationView {
  const record = asRecord(raw);
  if (!record) throw new TypeError("explain response is not a JSON object");
  const items: ExplainItem[] = [];
  if (Array.isArray(record.items)) {
    for (const item of record.items) {
      const parsed = itemOf(item);
      if (parsed) items.push(parsed);
    }
  }
  const conflictRows: string[] = [];
  if (Array.isArray(record.conflict_rows)) {
    for (const row of record.conflict_rows) {
      const name = asString(row);
      if (name) conflictRows.push(name);
    }
  }
  return {
    feasible: asBoolean(record.feasible),
    kind: asString(record.kind),
    proof: asString(record.proof),
    text: asString(record.text),
    totalRelaxation: asNumber(record.total_relaxation),
    conflictRows,
    farkas: farkasOf(record.farkas_check),
    items,
  };
}
