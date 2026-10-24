// The browser figures must reproduce the committed results:  node --test post/tests/sim.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { schedule } from "../js/sim.js";

// Python writes NaN (an undefined statistic) into results JSON; read it as null.
const read = (p) => JSON.parse(String(readFileSync(new URL(p, import.meta.url))).replace(/\bNaN\b/g, "null"));
const data = read("../data.json");
const frontier = read("../../results/frontier.json");
const future = read("../../results/future_only.json");
const sets = { musiccaps: data.prompts.filter((p) => p.set === "musiccaps"),
               redteam: data.prompts.filter((p) => p.set === "redteam") };
const close = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-9, `${msg}: js ${a} vs results ${b}`);

test("pruning schedules match results/frontier.json", () => {
  for (const s of frontier.schedules.filter((x) => x.first))
    for (const [name, ps] of Object.entries(sets)) {
      const js = schedule(ps, s.first, s.keep), py = s.sets[name];
      close(js.compute, py.compute.mean, `${s.first}/${s.keep} ${name} compute`);
      close(js.sameWinner, py.same_winner.mean, `${s.first}/${s.keep} ${name} same-winner`);
    }
});

test("leak-free kept rates match results/future_only.json", () => {
  for (const s of future.schedules)
    for (const [name, ps] of Object.entries(sets)) {
      const js = schedule(ps, s.first, s.keep), py = s.sets[name];
      assert.equal(js.testable, typeof py !== "string", `${s.first}/${s.keep} ${name} testable`);
      if (!js.testable) continue;
      close(js.kept, py.kept.mean, `${s.first}/${s.keep} ${name} kept`);
      close(js.chance, py.chance.mean, `${s.first}/${s.keep} ${name} chance`);
    }
});
