import { asBoolean, asRecord, asString, scalarRecord } from "./json.js";

export interface InfoView {
  version: string | null;
  clean: boolean | null;
  forbiddenPackages: string[];
  tripwireHits: number | null;
  tripwiresInstalled: boolean | null;
  scipyOptimizeImported: boolean | null;
  environment: [string, string][];
}

export function readInfo(raw: unknown): InfoView {
  const record = asRecord(raw);
  if (!record) throw new TypeError("info response is not a JSON object");
  const provenance = asRecord(record.provenance);
  const forbidden: string[] = [];
  if (Array.isArray(provenance?.forbidden_packages)) {
    for (const item of provenance.forbidden_packages) {
      const name = asString(item);
      if (name) forbidden.push(name);
    }
  }
  const hits = provenance?.tripwire_hits;
  const environment = Object.entries(scalarRecord(record.environment))
    .map(([key, value]): [string, string] => [key, value == null ? "—" : String(value)])
    .sort((a, b) => a[0].localeCompare(b[0]));
  return {
    version: asString(record.version),
    clean: asBoolean(provenance?.clean),
    forbiddenPackages: forbidden,
    tripwireHits: Array.isArray(hits) ? hits.length : null,
    tripwiresInstalled: asBoolean(provenance?.tripwires_installed),
    scipyOptimizeImported: asBoolean(provenance?.scipy_optimize_imported),
    environment,
  };
}
