import assert from "node:assert/strict";
import { test } from "node:test";
import { guessFormat } from "../static/js/lib/modeltext.js";
import { samples } from "../static/js/lib/sample.js";

test("format guess matches the server rule", () => {
  assert.equal(guessFormat(samples.diesel_week.text), "mps");
  assert.equal(guessFormat("Solve pooling using mip min cost .. x;"), "gams");
  assert.equal(guessFormat("Solve this later"), "mps");
});

test("bundled samples are MPS documents", () => {
  for (const sample of Object.values(samples)) {
    assert.match(sample.text, /^NAME /);
    assert.match(sample.text, /ENDATA\n$/);
    assert.ok(sample.label.endsWith(".mps"));
  }
});
