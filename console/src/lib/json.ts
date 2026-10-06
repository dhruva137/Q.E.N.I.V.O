export function asRecord(value: unknown): Record<string, unknown> | null {
  if (value == null || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

export function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

export function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function asBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

export function asPairs(value: unknown): [string, number][] {
  if (!Array.isArray(value)) return [];
  const out: [string, number][] = [];
  for (const item of value) {
    if (!Array.isArray(item) || item.length < 2) continue;
    const name = asString(item[0]);
    const number = asNumber(item[1]);
    if (name != null && number != null) out.push([name, number]);
  }
  return out;
}

export type Scalar = string | number | boolean | null;

export function scalarRecord(value: unknown): Record<string, Scalar> {
  const record = asRecord(value);
  const out: Record<string, Scalar> = {};
  if (!record) return out;
  for (const [key, item] of Object.entries(record)) {
    if (item == null) out[key] = null;
    else if (typeof item === "string" || typeof item === "boolean") out[key] = item;
    else if (typeof item === "number" && Number.isFinite(item)) out[key] = item;
  }
  return out;
}

export function numericRecord(value: unknown): Record<string, number | null> {
  const record = asRecord(value);
  const out: Record<string, number | null> = {};
  if (!record) return out;
  for (const [key, item] of Object.entries(record)) {
    if (item == null) out[key] = null;
    else if (typeof item === "number" && Number.isFinite(item)) out[key] = item;
  }
  return out;
}

export function countEntries(value: unknown): number | null {
  if (Array.isArray(value)) return value.length;
  const record = asRecord(value);
  return record ? Object.keys(record).length : null;
}
