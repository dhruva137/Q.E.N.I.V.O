export interface MarginalChange {
  name: string;
  left: number | null;
  right: number | null;
  delta: number;
}

export interface PlanSide {
  objective: number | null;
  marginals: readonly (readonly [string, number])[];
}

export interface ComparisonView {
  objectiveLeft: number | null;
  objectiveRight: number | null;
  objectiveChange: number | null;
  changes: MarginalChange[];
}

export function objectiveDelta(left: number | null, right: number | null): number | null {
  if (left == null || right == null) return null;
  if (!Number.isFinite(left) || !Number.isFinite(right)) return null;
  return right - left;
}

/** Rank row marginals by |right − left|. A name missing on one side contributes its full value. Exact zeros are omitted. */
export function marginalChanges(
  left: readonly (readonly [string, number])[],
  right: readonly (readonly [string, number])[],
  limit = 12,
): MarginalChange[] {
  const leftMap = new Map<string, number>();
  const rightMap = new Map<string, number>();
  for (const [name, value] of left) leftMap.set(name, value);
  for (const [name, value] of right) rightMap.set(name, value);
  const rows: MarginalChange[] = [];
  for (const name of new Set([...leftMap.keys(), ...rightMap.keys()])) {
    const a = leftMap.has(name) ? leftMap.get(name)! : null;
    const b = rightMap.has(name) ? rightMap.get(name)! : null;
    const delta = (b ?? 0) - (a ?? 0);
    if (delta === 0) continue;
    rows.push({ name, left: a, right: b, delta });
  }
  rows.sort((p, q) => Math.abs(q.delta) - Math.abs(p.delta) || p.name.localeCompare(q.name));
  return limit > 0 ? rows.slice(0, limit) : rows;
}

export function comparePlans(left: PlanSide, right: PlanSide, limit = 12): ComparisonView {
  return {
    objectiveLeft: left.objective,
    objectiveRight: right.objective,
    objectiveChange: objectiveDelta(left.objective, right.objective),
    changes: marginalChanges(left.marginals, right.marginals, limit),
  };
}
