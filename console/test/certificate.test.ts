import assert from "node:assert/strict";
import { test } from "node:test";
import { certificateSummary, parseCertificate, readVerifyReport } from "../static/js/lib/certificate.js";

const fixture = {
  schema: "qenivo.certificate/1",
  created_utc: "2026-09-30T09:00:00+00:00",
  verdict: "optimal",
  status: "optimal",
  tolerance: 1e-6,
  objective: 12.5,
  dual_objective: 12.5,
  residuals: { rel_primal: 1e-10, rel_dual: 2e-10, rel_gap: 3e-12, max_rel: 2e-10 },
  evidence: { kind: "kkt" },
  engine: { engine: "simplex", backend: "numpy", iterations: 4, time: 0.01, reason: "small LP" },
  environment: { qenivo: "0.1.0" },
  model: { name: "diesel_week", sha256: "abc", rows: 4, cols: 3, nnz: 7, sense: "maximize", source: null },
  solution: { x: { crude_light: 1, crude_heavy: 2, diesel: 3 }, y: { unit_capacity: 0.1 } },
};

test("parseCertificate reads verdict, residuals and engine", () => {
  const view = parseCertificate(fixture);
  assert.equal(view.verdict, "optimal");
  assert.equal(view.residuals.rel_primal, 1e-10);
  assert.equal(view.engine.engine, "simplex");
  assert.equal(view.engine.backend, "numpy");
  assert.equal(view.evidenceKind, "kkt");
  assert.equal(view.model.name, "diesel_week");
  assert.equal(view.solutionColumns, 3);
  assert.equal(view.solutionRows, 1);
  const rows = certificateSummary(view);
  assert.equal(rows.find(([key]) => key === "verdict")?.[1], "optimal");
  assert.equal(rows.find(([key]) => key === "residuals.rel_primal")?.[1], "1.000e-10");
  assert.equal(rows.find(([key]) => key === "engine.engine")?.[1], "simplex");
  assert.equal(rows.find(([key]) => key === "engine.reason")?.[1], "small LP");
});

test("parseCertificate accepts a JSON string and rejects a non-object", () => {
  assert.equal(parseCertificate('{"verdict":"infeasible","engine":{"engine":"simplex"}}').verdict, "infeasible");
  assert.throws(() => parseCertificate(null), /certificate is not a JSON object/);
  assert.throws(() => parseCertificate([]), /certificate is not a JSON object/);
});

test("readVerifyReport keeps each check name, flag and detail", () => {
  const view = readVerifyReport({
    passed: false,
    text: "rejected",
    model: "diesel_week",
    rows: 4,
    cols: 3,
    verdict: "optimal",
    tolerance: 1e-6,
    exact: false,
    checks: [
      ["model size matches certificate", true, "4x3 vs 4x3"],
      ["objective matches certificate", false, "1 vs 2"],
    ],
  });
  assert.equal(view.passed, false);
  assert.equal(view.checks.length, 2);
  assert.equal(view.checks[0].ok, true);
  assert.equal(view.checks[1].detail, "1 vs 2");
  assert.equal(view.exact, false);
});
