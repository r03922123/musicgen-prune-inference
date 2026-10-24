// The offline evaluation of torch/evaluate.py, ported line for line so the post's figures can
// recompute the committed results in the browser. post/tests/sim.test.mjs checks that they agree.

export const checkpoints = (first, horizon = 30) => {
  const out = [];
  for (let t = first; t < horizon; t += first) out.push(t);
  return out;
};

// Successive halving on one prompt. early: {t: scores of the t-second prefixes}; final: scores at
// the horizon. Ties keep lower row ids, as Python's stable sort and max do.
export function simulate(early, final, cps, keepFrac, horizon = 30) {
  const n = final.length;
  let alive = [...Array(n).keys()], spent = 0, prev = 0;
  const log = {};
  for (const t of cps) {
    if (alive.length === 1) break;
    spent += alive.length * (t - prev);
    const keep = Math.max(1, Math.ceil(alive.length * keepFrac));
    const order = [...alive].sort((a, b) => early[t][b] - early[t][a] || a - b);
    alive = order.slice(0, keep).sort((a, b) => a - b);
    log[t] = alive;
    prev = t;
  }
  spent += alive.length * (horizon - prev);
  const winner = alive.reduce((w, r) => (final[r] > final[w] ? r : w), alive[0]);
  return { alive: log, survivors: alive, winner, compute: spent / (n * horizon) };
}

const argmax = (x) => x.reduce((b, v, i) => (v > x[b] ? i : b), 0);
const mean = (x) => x.reduce((a, b) => a + b, 0) / x.length;
const early = (p) => Object.fromEntries(Object.entries(p.clap).map(([t, v]) => [Number(t), v]));

// One prompt: does pruning keep the take unpruned best-of-8 would pick? (and the leak-free check's target)
export function prune(p, first, keep) {
  const e = early(p), cps = first ? checkpoints(first) : [];
  const r = simulate(e, e[30], cps, keep);
  const base = argmax(e[30]);
  const target = argmax(p.chunks.map((c) => c[2]));        // best on the unheard 20-30 s chunk
  const testable = !Object.keys(r.alive).some((t) => Number(t) > 20);
  return { ...r, cps, base, same: r.winner === base, target, testable,
           kept: r.survivors.includes(target), chance: r.survivors.length / p.n };
}

// Averages over a set of prompts, as frontier.json and future_only.json report them.
export function schedule(prompts, first, keep) {
  const rs = prompts.map((p) => prune(p, first, keep));
  const testable = rs.every((r) => r.testable);
  return {
    n: rs.length,
    compute: mean(rs.map((r) => r.compute)),
    sameWinner: mean(rs.map((r) => +r.same)),
    testable,
    kept: testable ? mean(rs.map((r) => +r.kept)) : NaN,
    chance: testable ? mean(rs.map((r) => r.chance)) : NaN,
  };
}
