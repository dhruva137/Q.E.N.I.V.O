import assert from "node:assert/strict";
import { test } from "node:test";
import { comparePlans, marginalChanges, objectiveDelta } from "../static/js/lib/compare.js";

test("objective change is plan B minus plan A", () => {
  assert.equal(objectiveDelta(100, 130), 30);
  assert.equal(objectiveDelta(null, 1), null);
  assert.equal(objectiveDelta(1, Number.NaN), null);
});

test("largest marginal changes are ranked by absolute difference", () => {
  const left: [string, number][] = [
    ["diesel_contract", 10],
    ["sulphur_spec", 1],
    ["unit_capacity", -2],
  ];
  const right: [string, number][] = [
    ["diesel_contract", 10],
    ["sulphur_spec", 4],
    ["unit_capacity", -8],
  ];
  const changes = marginalChanges(left, right);
  assert.deepEqual(changes.map((row) => row.name), ["unit_capacity", "sulphur_spec"]);
  assert.equal(changes[0].delta, -6);
  assert.equal(changes[1].delta, 3);
});

test("a marginal present on only one plan is an absence on the other", () => {
  const changes = marginalChanges([["only_a", 5]], [["only_b", -1]]);
  assert.deepEqual(changes.map((row) => [row.name, row.left, row.right, row.delta]), [
    ["only_a", 5, null, -5],
    ["only_b", null, -1, -1],
  ]);
});

test("equal absolute changes sort by row name and honor the limit", () => {
  const changes = marginalChanges([["b", 0], ["a", 1]], [["b", 1], ["a", 2]], 1);
  assert.equal(changes.length, 1);
  assert.equal(changes[0].name, "a");
  assert.equal(changes[0].delta, 1);
});

test("comparePlans reports both objectives and the ranked marginals", () => {
  const view = comparePlans(
    { objective: 100, marginals: [["cap", 1]] },
    { objective: 80, marginals: [["cap", 4]] },
  );
  assert.equal(view.objectiveChange, -20);
  assert.equal(view.changes[0].name, "cap");
  assert.equal(view.changes[0].delta, 3);
});
