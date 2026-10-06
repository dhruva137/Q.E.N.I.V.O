export function formatNumber(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value !== 0 && (Math.abs(value) >= 1e7 || Math.abs(value) < 1e-3)) return value.toExponential(3);
  const rounded = Math.round(value * 1e6) / 1e6;
  if (rounded === 0) return "0";
  return String(rounded);
}

export function formatScalar(value: string | number | boolean | null | undefined): string {
  if (value == null) return "—";
  if (typeof value === "number") return formatNumber(value);
  return String(value);
}
