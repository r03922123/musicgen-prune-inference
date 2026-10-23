// The browser figures must reproduce the committed results:  node --test post/tests/sim.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { agreement, bestOfK, meanRho, schedule } from "../js/sim.js";

// Python writes NaN (an undefined statistic) into results JSON; read it as null.
const read = (p) => JSON.parse(String(readFileSync(new URL(p, import.meta.url))).replace(/\bNaN\b/g, "null"));
const data = read("../data.json");
const frontier = read("../../results/frontier.json");
const future = read("../../results/future_only.json");
const agree = read("../../results/judge_agreement.json");
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

test("best-of-8 by CLAP and by the judge match the Q2 selection gains", () => {
  for (const [judge, sel] of Object.entries(frontier.selection)) {
    const ps = sets.musiccaps;
    const avg = (by) => ps.reduce((a, p) => a + bestOfK(p, judge, 8, by), 0) / ps.length;
    close(avg("clap"), sel.musiccaps.gain_vs_random.mean, `${judge} CLAP pick gain`);
    close(avg("judge"), sel.musiccaps.oracle_gain.mean, `${judge} oracle gain`);
  }
});

test("10 s score vs heard and unheard chunks matches the predictability check", () => {
  for (const [name, ps] of Object.entries(sets)) {
    const p = future.predictability[name];
    close(meanRho(ps, "heard"), p.clap10_vs_chunk1.mean, `${name} heard`);
    close(meanRho(ps, "unheard"), p.clap10_vs_chunks23.mean, `${name} unheard`);
  }
});

test("judge agreement matches results/judge_agreement.json", () => {
  for (const [pair, r] of Object.entries(agree.pairs)) {
    const [a, b] = pair.split("~"), js = agreement(data.prompts, a, b);
    close(js.within, r.within.mean, `${pair} within`);
    close(js.between, r.between, `${pair} between`);
    close(js.topMatch, r.top_match.mean, `${pair} top match`);
  }
});
