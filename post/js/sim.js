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

// Q1 for one prompt: does pruning keep the take unpruned best-of-n would pick?
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

// Best-of-k inside the 8 takes: average over every k-subset of judge(pick) - mean judge(all 8),
// where the pick is the subset's best by `by` ("judge" itself = oracle, or "clap" = CLAP at 30 s).
export function bestOfK(p, judge, k, by) {
  const j = p.judges[judge], sel = by === "clap" ? p.clap["30"] : j, m = mean(j);
  let total = 0, count = 0;
  const rec = (start, chosen) => {
    if (chosen.length === k) {
      const pick = chosen.reduce((b, i) => (sel[i] > sel[b] ? i : b), chosen[0]);
      total += j[pick] - m; count += 1; return;
    }
    for (let i = start; i < j.length; i++) rec(i + 1, [...chosen, i]);
  };
  rec(0, []);
  return total / count;
}

// Spearman's rho with average ranks for ties (scipy's default); NaN if either side is constant.
function ranks(x) {
  const idx = x.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(x.length);
  for (let i = 0; i < idx.length;) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}
export function spearman(x, y) {
  const a = ranks(x), b = ranks(y), ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) {
    sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2;
  }
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : NaN;
}

// Within-prompt agreement of the 10 s score with a chunk target, averaged over prompts (NaNs dropped).
export const chunkTarget = (p, which) =>
  which === "heard" ? p.chunks.map((c) => c[0]) : p.chunks.map((c) => (c[1] + c[2]) / 2);
export function meanRho(prompts, which) {
  const v = prompts.map((p) => spearman(p.clap["10"], chunkTarget(p, which))).filter((x) => !Number.isNaN(x));
  return mean(v);
}
