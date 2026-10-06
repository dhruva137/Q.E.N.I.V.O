import { asNumber, asRecord, asString } from "./json.js";

/** One line of audit-YYYY-MM.jsonl. utc is added by AccessPolicy.audit; the other keys are filled in server.Handler. */
export interface AuditRecord {
  utc: string | null;
  event: string | null;
  path: string | null;
  user: string | null;
  auth: string | null;
  client: string | null;
  model_sha256: string | null;
  engine_requested: string | null;
  status: number | null;
  verdict: string | null;
  objective: number | null;
  engine: string | null;
  seconds: number | null;
  error: string | null;
  reason: string | null;
}

export interface AuditLog {
  records: AuditRecord[];
  skipped: number;
}

function recordOf(value: unknown): AuditRecord | null {
  const record = asRecord(value);
  if (!record) return null;
  return {
    utc: asString(record.utc),
    event: asString(record.event),
    path: asString(record.path),
    user: asString(record.user),
    auth: asString(record.auth),
    client: asString(record.client),
    model_sha256: asString(record.model_sha256),
    engine_requested: asString(record.engine_requested),
    status: asNumber(record.status),
    verdict: asString(record.verdict),
    objective: asNumber(record.objective),
    engine: asString(record.engine),
    seconds: asNumber(record.seconds),
    error: asString(record.error),
    reason: asString(record.reason),
  };
}

export function parseAuditLog(text: string): AuditLog {
  let body = text;
  if (body.charCodeAt(0) === 0xfeff) body = body.slice(1);
  const records: AuditRecord[] = [];
  let skipped = 0;
  for (const line of body.split(/\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const parsed = recordOf(JSON.parse(trimmed) as unknown);
      if (parsed) records.push(parsed);
      else skipped += 1;
    } catch {
      skipped += 1;
    }
  }
  return { records, skipped };
}
