import assert from "node:assert/strict";
import { test } from "node:test";
import { presentExplanation } from "../static/js/lib/explain.js";

test("a conflict explanation keeps items, Farkas fields and text", () => {
  const view = presentExplanation({
    feasible: false,
    kind: "conflict",
    items: [{
      constraint: "need",
      issue: "cannot reach lower limit 5; best is 3 (short by 2)",
      limit: "lower",
      limit_value: 5,
      best_achievable: 3,
      shortfall: 2,
    }],
    total_relaxation: 2,
    conflict_rows: ["need", "cap"],
    farkas_check: { valid: true, value: 1, violation: 0, ratio: 0 },
    proof: "elastic LP optimum > 0; its row duals are a Farkas ray (checked)",
    text: "The model is INFEASIBLE.",
  });
  assert.equal(view.feasible, false);
  assert.equal(view.kind, "conflict");
  assert.equal(view.items[0].constraint, "need");
  assert.equal(view.items[0].shortfall, 2);
  assert.equal(view.items[0].bestAchievable, 3);
  assert.deepEqual(view.conflictRows, ["need", "cap"]);
  assert.equal(view.farkas?.valid, true);
  assert.equal(view.totalRelaxation, 2);
  assert.match(view.text ?? "", /INFEASIBLE/);
});

test("a bound conflict has no shortfall and a feasible model has no items", () => {
  const conflict = presentExplanation({
    feasible: false,
    kind: "bound_conflict",
    items: [{ constraint: "x", issue: "lower bound 5 > upper bound 3" }],
    proof: "direct: contradictory bounds on one column",
  });
  assert.equal(conflict.items[0].shortfall, null);
  assert.equal(conflict.items[0].issue, "lower bound 5 > upper bound 3");
  assert.equal(conflict.farkas, null);
  const feasible = presentExplanation({ feasible: true, kind: "feasible", items: [], total_relaxation: 0, proof: "" });
  assert.equal(feasible.feasible, true);
  assert.equal(feasible.items.length, 0);
  assert.equal(feasible.totalRelaxation, 0);
});
