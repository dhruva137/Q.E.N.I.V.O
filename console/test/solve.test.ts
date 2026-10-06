import assert from "node:assert/strict";
import { test } from "node:test";
import { readSolve } from "../static/js/lib/solve.js";

test("an LP solve response keeps verdict, tables, residuals and certificate", () => {
  const certificate = { schema: "qenivo.certificate/1", verdict: "optimal" };
  const view = readSolve({
    kind: "lp",
    model: "diesel_week",
    size: "4 x 3",
    verdict: "optimal",
    objective: 42,
    residuals: { rel_primal: 1e-12, rel_dual: 0, rel_gap: 0, max_rel: 1e-12 },
    engine: { engine: "simplex", backend: "numpy", iterations: 3, time: 0.004, reason: "small" },
    top_values: [["diesel", 10], ["crude_light", 4]],
    marginals: [["diesel_contract", 130], ["unit_capacity", -2]],
    certificate,
  });
  assert.equal(view.verdict, "optimal");
  assert.equal(view.objective, 42);
  assert.equal(view.engine.engine, "simplex");
  assert.deepEqual(view.topValues[0], ["diesel", 10]);
  assert.deepEqual(view.marginals[0], ["diesel_contract", 130]);
  assert.equal(view.iterations, 3);
  assert.equal(view.certificate, certificate);
  assert.equal(view.residuals.max_rel, 1e-12);
});

test("a bilinear solve response uses the top-level status fields", () => {
  const view = readSolve({
    kind: "bilinear",
    size: "pools",
    objective: 1.25,
    status: "optimal",
    feasible: true,
    max_violation: 1e-8,
    iterations: 6,
    time: 0.2,
    history: [{ iter: 1 }, { iter: 2 }],
  });
  assert.equal(view.kind, "bilinear");
  assert.equal(view.feasible, true);
  assert.equal(view.maxViolation, 1e-8);
  assert.equal(view.iterations, 6);
  assert.equal(view.historyPoints, 2);
  assert.equal(view.marginals.length, 0);
  assert.equal(view.verdict, null);
});

test("an infeasible solve keeps the embedded explanation proof", () => {
  const view = readSolve({
    kind: "lp",
    verdict: "infeasible",
    objective: null,
    explanation: { proof: "direct: contradictory bounds on one column", feasible: false, kind: "bound_conflict" },
  });
  assert.equal(view.verdict, "infeasible");
  assert.equal(view.explanationProof, "direct: contradictory bounds on one column");
  assert.equal(view.objective, null);
});
