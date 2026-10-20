// Pure game logic, no DOM. Mirrors torch/evaluate.py (all_brackets, bracket_champion) so the
// machine player in the browser is the same algorithm as results/bracket.json.

export const ROUND_SECONDS = [10, 20, 30];
export const ROUND_NAMES = ["Quarter-finals", "Semi-finals", "Final"];

// Seeded PRNG (mulberry32) so a game can be replayed from its seed.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffle(xs, rand) {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Higher score wins; ties go to the lower row id (as in evaluate.bracket_champion).
export function better(a, b, scores) {
  return scores[a] > scores[b] || (scores[a] === scores[b] && a < b) ? a : b;
}

// rounds[i] = scores used in round i. Returns the survivors after each round.
export function playOut(seeding, rounds) {
  const stages = [];
  let alive = [...seeding];
  for (const s of rounds) {
    const next = [];
    for (let i = 0; i < alive.length; i += 2) next.push(better(alive[i], alive[i + 1], s));
    stages.push(next);
    alive = next;
  }
  return stages;
}

export function allBrackets(rows) {
  if (typeof rows === "number") rows = [...Array(rows).keys()];
  if (rows.length === 1) return [rows];
  const [first, ...rest] = rows;
  const half = rows.length / 2;
  const out = [];
  const pick = (start, chosen) => {
    if (chosen.length === half - 1) {
      const a = [first, ...chosen];
      const b = rest.filter((r) => !chosen.includes(r));
      for (const x of allBrackets(a)) for (const y of allBrackets(b)) out.push([...x, ...y]);
      return;
    }
    for (let i = start; i < rest.length; i++) pick(i + 1, [...chosen, rest[i]]);
  };
  pick(0, []);
  return out;
}

export function clapRounds(prompt) {
  return ROUND_SECONDS.map((t) => prompt.clap[String(t)]);
}

// Rank of every take under a score list: 1 = best (ties share the better rank).
export function ranks(scores) {
  return scores.map((s) => 1 + scores.filter((x) => x > s).length);
}

export const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0);

// The player's matches: [{round, a, b, winner, tie}]. For each, would CLAP pick the same take
// from that same pair at that clip length?
export function agreement(matches, prompt) {
  const rounds = clapRounds(prompt);
  const per = matches.map((m) => better(m.a, m.b, rounds[m.round]) === m.winner);
  return { agree: per.filter(Boolean).length, total: per.length, per };
}

// Listening seconds: restart-from-0 bracket vs hearing every take in full.
export function listeningPlan(n = 8, seconds = ROUND_SECONDS, horizon = 30) {
  let alive = n, restart = 0, cont = 0, prev = 0;
  for (const t of seconds) {
    restart += alive * t;
    cont += alive * (t - prev);
    prev = t;
    alive /= 2;
  }
  return { restart, continued: cont, full: n * horizon };
}
