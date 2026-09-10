// Run with: node --test site/tests/   (built-in test runner, no dependencies)
import assert from "node:assert/strict";
import { test } from "node:test";

import { extent, fmt, inkOn, linear, nearestIndex, niceTicks, seqColor } from "../assets/charts.js";

test("linear maps and inverts", () => {
  const f = linear(1900, 1960, 0, 600);
  assert.equal(f(1930), 300);
  assert.equal(f.invert(150), 1915);
});

test("niceTicks produces clean, covering steps", () => {
  assert.deepEqual(niceTicks(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  assert.deepEqual(niceTicks(1900, 1963, 6), [1900, 1910, 1920, 1930, 1940, 1950, 1960]);
  assert.deepEqual(niceTicks(3, 3), [3]);
});

test("extent ignores missing values and pads", () => {
  assert.deepEqual(extent([null, 1, NaN, 3]), [1, 3]);
  const [lo, hi] = extent([0, 10], 0.1);
  assert.ok(Math.abs(lo + 1) < 1e-9 && Math.abs(hi - 11) < 1e-9);
  assert.deepEqual(extent([]), [0, 1]);
});

test("nearestIndex picks the closest x", () => {
  assert.equal(nearestIndex([1900, 1903, 1906], 1904.9), 2);
  assert.equal(nearestIndex([1900, 1903, 1906], 1901), 0);
});

test("formatters never print NaN and use a true minus", () => {
  assert.equal(fmt.pct(null), "—");
  assert.equal(fmt.pct(0.1234), "12.3%");
  assert.equal(fmt.pp(-1.5), "−1.50 pp");
  assert.equal(fmt.pp(2), "+2.00 pp");
  assert.equal(fmt.num(12345.6), "12,346");
});

test("sequential ramp is clamped and cell ink follows luminance", () => {
  assert.equal(seqColor(-1), "#cde2fb");
  assert.equal(seqColor(2), "#0d366b");
  assert.match(inkOn("#cde2fb"), /dark/);
  assert.match(inkOn("#0d366b"), /light/);
});
