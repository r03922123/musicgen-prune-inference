// node --test site/tests
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { allBrackets, playOut, agreement, listeningPlan, ranks, shuffle, rng, clapRounds, better, argmax } from "../bracket.js";

const data = JSON.parse(readFileSync(new URL("../data.json", import.meta.url)));
const bracketJson = JSON.parse(readFileSync(new URL("../../results/bracket.json", import.meta.url)));

test("bracket counts match n!/2^(n-1)", () => {
  assert.equal(allBrackets(2).length, 1);
  assert.equal(allBrackets(4).length, 3);
  assert.equal(allBrackets(8).length, 315);
});

test("toy bracket as in test_evaluate.py", () => {
  const r = [[.4, .3, .2, .1], [.1, .2, .9, .3]];
  assert.equal(playOut([0, 1, 2, 3], r).at(-1)[0], 2);
  assert.equal(playOut([0, 3, 1, 2], r).at(-1)[0], 1);
  assert.equal(better(3, 2, [.5, .5, .5, .5]), 2);
});

test("machine player reproduces results/bracket.json (same_as_unpruned, every set)", () => {
  for (const set of ["musiccaps", "redteam"]) {
    const ps = data.prompts.filter((p) => p.set === set);
    const shares = ps.map((p) => {
      const rounds = clapRounds(p), base = argmax(p.clap["30"]);
      const bs = allBrackets(8);
      return bs.filter((b) => playOut(b, rounds).at(-1)[0] === base).length / bs.length;
    });
    const mean = shares.reduce((a, b) => a + b, 0) / shares.length;
    const py = bracketJson.judges.cmi_qual[set].same_as_unpruned;
    assert.equal(shares.length, py.n);
    assert.ok(Math.abs(mean - py.mean) < 1e-12, `${set}: js ${mean} vs py ${py.mean}`);
  }
});

test("agreement counts the player's matches CLAP would decide the same way", () => {
  const p = { clap: { 10: [8, 7, 6, 5, 4, 3, 2, 1], 20: [8, 7, 6, 5, 4, 3, 2, 1], 30: [1, 2, 3, 4, 5, 6, 7, 8] } };
  const m = [{ round: 0, a: 0, b: 1, winner: 0 }, { round: 0, a: 2, b: 3, winner: 3 }, { round: 2, a: 0, b: 7, winner: 0 }];
  assert.deepEqual(agreement(m, p), { agree: 1, total: 3, per: [true, false, false] });
});

test("listening plan: restart 220 s, continue 140 s, full 240 s", () => {
  assert.deepEqual(listeningPlan(), { restart: 220, continued: 140, full: 240 });
});

test("ranks and seeded shuffle", () => {
  assert.deepEqual(ranks([0.1, 0.9, 0.5, 0.9]), [4, 1, 3, 1]);
  assert.deepEqual(shuffle([0, 1, 2, 3, 4, 5, 6, 7], rng(1)), shuffle([0, 1, 2, 3, 4, 5, 6, 7], rng(1)));
  assert.deepEqual([...shuffle([0, 1, 2, 3, 4, 5, 6, 7], rng(1))].sort(), [0, 1, 2, 3, 4, 5, 6, 7]);
});
