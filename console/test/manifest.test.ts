import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { library } from "../static/js/lib/manifest.js";

const industryPath = fileURLToPath(new URL("../../src/qenivo/models/industry.py", import.meta.url));

test("the manifest lists industry.py LIBRARY in order", () => {
  const text = fs.readFileSync(industryPath, "utf8").replace(/\r\n/g, "\n");
  const block = text.match(/\nLIBRARY\s*=\s*\{([\s\S]*?)\n\}/);
  assert.ok(block);
  const names = [...block[1].matchAll(/"([A-Za-z_][A-Za-z0-9_]*)"\s*:/g)].map((match) => match[1]);
  assert.deepEqual(library.map((model) => model.name), names);
  assert.ok(names.length >= 8);
  for (const model of library) {
    assert.ok(model.summary.length > 0);
    assert.equal(model.detail.split("\n")[0].trim(), model.summary);
    assert.match(model.problemClass, /^(LP|QP|MILP)$/);
    assert.ok(model.sector.length > 0);
    assert.ok(model.formulation.length > 0);
    assert.match(model.command, new RegExp(`^qenivo model ${model.name} `));
    assert.match(model.command, new RegExp(`-o ${model.name}\\.mps$`));
    for (const parameter of model.parameters) {
      assert.match(model.command, new RegExp(`-p ${parameter.name}=${parameter.default}( |$)`));
    }
  }
});
