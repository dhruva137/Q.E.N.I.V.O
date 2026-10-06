import assert from "node:assert/strict";
import { test } from "node:test";
import { readInfo } from "../static/js/lib/info.js";

test("readInfo keeps version, provenance and environment", () => {
  const view = readInfo({
    version: "0.4.0",
    environment: { qenivo: "0.4.0", python: "3.13.11", numpy: "2.4.4", gpu: "none" },
    provenance: {
      clean: true,
      forbidden_packages: [],
      tripwire_hits: [],
      tripwires_installed: true,
      scipy_optimize_imported: false,
    },
  });
  assert.equal(view.version, "0.4.0");
  assert.equal(view.clean, true);
  assert.deepEqual(view.forbiddenPackages, []);
  assert.equal(view.tripwireHits, 0);
  assert.equal(view.tripwiresInstalled, true);
  assert.equal(view.scipyOptimizeImported, false);
  assert.equal(view.environment.find(([key]) => key === "qenivo")?.[1], "0.4.0");
});

test("a provenance violation lists the forbidden packages", () => {
  const view = readInfo({
    version: "0.4.0",
    environment: {},
    provenance: { clean: false, forbidden_packages: ["gurobi"], tripwire_hits: [{ call: "linprog" }] },
  });
  assert.equal(view.clean, false);
  assert.deepEqual(view.forbiddenPackages, ["gurobi"]);
  assert.equal(view.tripwireHits, 1);
});
