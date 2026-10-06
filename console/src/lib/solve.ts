import { asBoolean, asNumber, asPairs, asRecord, asString, numericRecord, scalarRecord } from "./json.js";

export interface SolveView {
  kind: string;
  model: string | null;
  size: string | null;
  verdict: string | null;
  objective: number | null;
  residuals: Record<string, number | null>;
  engine: Record<string, string | number | boolean | null>;
  topValues: [string, number][];
  marginals: [string, number][];
  certificate: unknown | null;
  feasible: boolean | null;
  status: string | null;
  maxViolation: number | null;
  iterations: number | null;
  time: number | null;
  historyPoints: number | null;
  explanationProof: string | null;
}

export function readSolve(raw: unknown): SolveView {
  const record = asRecord(raw);
  if (!record) throw new TypeError("solve response is not a JSON object");
  const engine = asRecord(record.engine);
  const explanation = asRecord(record.explanation);
  const certificate = record.certificate;
  return {
    kind: asString(record.kind) ?? "lp",
    model: asString(record.model),
    size: asString(record.size),
    verdict: asString(record.verdict),
    objective: asNumber(record.objective),
    residuals: numericRecord(record.residuals),
    engine: scalarRecord(record.engine),
    topValues: asPairs(record.top_values),
    marginals: asPairs(record.marginals),
    certificate: certificate == null ? null : certificate,
    feasible: asBoolean(record.feasible),
    status: asString(record.status),
    maxViolation: asNumber(record.max_violation),
    iterations: asNumber(record.iterations) ?? asNumber(engine?.iterations),
    time: asNumber(record.time) ?? asNumber(engine?.time),
    historyPoints: Array.isArray(record.history) ? record.history.length : null,
    explanationProof: asString(explanation?.proof),
  };
}
