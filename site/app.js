import {
  ROUND_SECONDS, ROUND_NAMES, rng, shuffle, playOut, clapRounds, ranks, argmax, agreement, listeningPlan,
} from "./bracket.js";

const LETTERS = "ABCDEFGH";
const CRITICS = {
  aes_CE: { name: "The Vibe", model: "Audiobox Aesthetics · content enjoyment", year: "Meta, 2025",
            measures: "how enjoyable the music is, ignoring the text" },
  mulan: { name: "The Literalist", model: "MuQ-MuLan", year: "2025",
           measures: "how closely the music matches the prompt's words" },
  cmi_qual: { name: "The Newcomer · quality", model: "CMI-RM", year: "2026",
              measures: "overall musical quality, from a reward model trained on human preferences" },
  cmi_ali: { name: "The Newcomer · match", model: "CMI-RM", year: "2026",
             measures: "how well the music fits the prompt, from the same reward model" },
};
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const fmtT = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const pct = (x) => `${Math.round(x * 100)}%`;
const sgnless = (x, d = 2) => `${x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)}`;
const IOS = /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
// Tiers instead of exact ranks: neighbouring takes are often within a critic's noise.
const tier = (r) => (r <= 2 ? "top" : r >= 7 ? "bottom" : "middle");
const TIER_LABEL = { top: "Top two", middle: "Middle", bottom: "Bottom two" };
const sgn = (x, d = 2) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;

let DATA, G = null;           // G = current game

// ---------- storage (best effort; the game works without it) ----------
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode etc. */ } },
};

// ---------- covers ----------
function hash(s) { let h = 2166136261; for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return h >>> 0; }
function gradient(key) {
  const h = hash(key), a = h % 360, b = (a + 40 + (h >> 9) % 80) % 360;
  return `linear-gradient(${(h >> 3) % 360}deg, hsl(${a} 70% 52%), hsl(${b} 65% 30%))`;
}
function cover(el, key, label = "") { el.style.setProperty("--g", gradient(key)); el.textContent = label; return el; }
function coverEl(key, label, cls = "") {
  const d = document.createElement("div"); d.className = `cover ${cls}`; d.setAttribute("aria-hidden", "true");
  return cover(d, key, label);
}

// ---------- audio (Web Audio: exact clip length, short fades, cached buffers) ----------
const audio = {
  ctx: null, cache: new Map(), cur: null,
  ensure() { this.ctx ??= new (window.AudioContext || window.webkitAudioContext)(); if (this.ctx.state === "suspended") this.ctx.resume(); },
  load(url) {
    if (!this.cache.has(url)) {
      this.ensure();
      this.cache.set(url, fetch(url).then((r) => { if (!r.ok) throw new Error(url); return r.arrayBuffer(); })
        .then((b) => this.ctx.decodeAudioData(b)));
    }
    return this.cache.get(url);
  },
  stop() {
    if (!this.cur) return;
    const c = this.cur; this.cur = null;
    try { c.src.onended = null; c.src.stop(); } catch { /* already stopped */ }
    cancelAnimationFrame(c.raf); c.done?.(false);
  },
  // Plays [0, seconds) of url. onTick(elapsed, total); resolves true if it played to the end.
  async play(url, seconds, { onTick, title, sub, key } = {}) {
    this.stop(); this.ensure();
    nowPlaying(title, sub, key, 0, seconds);
    const buf = await this.load(url);
    const dur = Math.min(seconds, buf.duration);
    return new Promise((resolve) => {
      const src = this.ctx.createBufferSource(), g = this.ctx.createGain();
      src.buffer = buf; src.connect(g).connect(this.ctx.destination);
      const t0 = this.ctx.currentTime, fade = Math.min(0.4, dur / 4);
      g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(1, t0 + 0.02);
      g.gain.setValueAtTime(1, t0 + dur - fade); g.gain.linearRampToValueAtTime(0, t0 + dur);
      const c = { src, done: resolve, raf: 0 };
      const tick = () => {
        const e = Math.min(this.ctx.currentTime - t0, dur);
        onTick?.(e, dur); nowPlaying(title, sub, key, e, dur);
        if (this.cur === c) c.raf = requestAnimationFrame(tick);
      };
      src.onended = () => { if (this.cur === c) { this.cur = null; cancelAnimationFrame(c.raf); onTick?.(dur, dur); nowPlaying(title, sub, key, dur, dur, true); resolve(true); } };
      this.cur = c; src.start(t0, 0, dur); tick();
    });
  },
};
const url = (pid, row) => `audio/${pid}_${row}.mp3`;

function nowPlaying(title, sub, key, e, total, ended = false) {
  $("#np-title").textContent = title ?? "Nothing playing";
  $("#np-sub").textContent = sub ?? "";
  if (key) cover($("#np-cover"), key, title?.slice(-1) ?? "");
  $("#np-fill").style.width = total ? `${(100 * e) / total}%` : "0";
  $("#np-time").textContent = `${fmtT(e)} / ${fmtT(total || 0)}`;
  $("#np-toggle").disabled = ended || !audio.cur;
}

// ---------- views ----------
function show(id) {
  $$(".view").forEach((v) => (v.hidden = v.id !== id));
  window.scrollTo({ top: 0, behavior: "instant" });
}

function renderShelf(filter = "all") {
  const shelf = $("#shelf"); shelf.innerHTML = "";
  for (const p of DATA.prompts.filter((p) => filter === "all" || p.set === filter)) {
    const b = document.createElement("button"); b.className = "tile";
    b.append(coverEl(p.id, "", ""));
    const t = document.createElement("b"); t.textContent = p.title;
    const tag = document.createElement("span");
    tag.className = `tag ${p.set === "redteam" ? "slow" : ""}`;
    tag.textContent = p.set === "redteam" ? "Slow build" : "MusicCaps caption";
    b.append(t, tag); b.title = p.caption;
    b.onclick = () => startGame(p);
    shelf.append(b);
  }
}

// ---------- game ----------
function startGame(prompt) {
  audio.stop();
  audio.cache.clear();                           // decoded takes are ~4 MB each; keep one prompt's worth
  $("#caption").open = true;
  $("#ios-hint").hidden = !IOS;
  if (!G?.active) history.pushState({ game: true }, "", "#play");
  const seed = (Date.now() ^ (Math.random() * 2 ** 32)) >>> 0;
  const rand = rng(seed);
  const seeding = shuffle([...Array(prompt.n).keys()], rand);
  G = { prompt, seed, rand, seeding, round: 0, match: 0, alive: seeding, stages: [], matches: [],
        listened: 0, started: Date.now(), active: true };
  G.letter = (row) => LETTERS[seeding.indexOf(row)];
  G.key = (row) => `${prompt.id}:${G.letter(row)}`;
  $("#prompt-title").textContent = prompt.title;
  $("#prompt-caption").textContent = prompt.caption;
  // warm the cache for round 1
  seeding.forEach((r) => audio.load(url(prompt.id, r)).catch(() => {}));
  show("game"); renderMatch();
}

function renderBracket(el, stages, seeding, { cur = [], letter, key }) {
  el.innerHTML = "";
  el.setAttribute("role", "img");
  el.setAttribute("aria-label", stages.length
    ? stages.map((st, i) => `${ROUND_NAMES[i]}: ${st.map((r) => `Take ${letter(r)}`).join(", ")} advanced`).join(". ")
    : `Eight takes: ${seeding.map((r) => letter(r)).join(", ")}`);
  const cols = [seeding, ...stages];
  while (cols.length < 4) cols.push(Array(cols.at(-1).length / 2 || 1).fill(null));
  cols.forEach((col, ci) => {
    const c = document.createElement("div"); c.className = "col";
    col.forEach((row) => {
      const s = document.createElement("div");
      if (row === null) { s.className = "slot empty"; c.append(s); return; }
      const out = ci < stages.length && !stages[ci].includes(row);
      s.className = `slot${out ? " out" : ""}${cur.includes(row) && ci === stages.length ? " cur" : ""}`;
      s.append(coverEl(key(row), letter(row)), document.createTextNode(`Take ${letter(row)}`));
      c.append(s);
    });
    el.append(c);
  });
}

function renderMatch() {
  const r = G.round, sec = ROUND_SECONDS[r], nMatches = G.alive.length / 2;
  const a = G.alive[2 * G.match], b = G.alive[2 * G.match + 1];
  const [left, right] = G.rand() < 0.5 ? [a, b] : [b, a];
  G.cur = { a, b, left, right, heard: { a: false, b: false }, firstPlayed: null, replays: 0, t0: Date.now(), decided: false };
  renderProgress();
  $("#round-label").textContent = ROUND_NAMES[r];
  $("#match-label").textContent = nMatches > 1 ? `Match ${G.match + 1} of ${nMatches}` : "The final";
  $("#clip-label").textContent = sec === 30 ? "Full 30 s" : `${sec} s clips`;
  for (const [side, row] of [["a", left], ["b", right]]) {
    const el = $(`#take-${side}`);
    el.classList.remove("playing", "heard", "won", "lost");
    $(".tid", el).textContent = G.letter(row);
    cover($(".cover", el), G.key(row), G.letter(row));
    $(".bar i", el).style.width = "0";
    $(".heard-sr", el).textContent = "";
  }
  $$(".pick").forEach((p) => (p.disabled = true)); $("#tie").disabled = true;
  $("#hint").textContent = r === 0
    ? "Listen to both takes, then keep the one you'd rather keep listening to."
    : `Clips are longer now (${sec} s), so you'll hear where each take goes.`;
}

function renderProgress() {
  const done = G.matches.length, el = $("#progress");
  el.innerHTML = "";
  [4, 2, 1].forEach((n, r) => {
    if (r) el.insertAdjacentHTML("beforeend", '<li class="gap" aria-hidden="true"></li>');
    for (let i = 0; i < n; i++) {
      const k = [0, 4, 6][r] + i;
      el.insertAdjacentHTML("beforeend", `<li class="${k < done ? "done" : k === done ? "cur" : ""}" aria-hidden="true"></li>`);
    }
  });
  el.setAttribute("aria-label", `Match ${done + 1} of 7 overall`);
}

async function playSide(side) {
  if (!G?.cur || G.cur.decided) return false;
  const m = G.cur, row = side === "a" ? m.left : m.right, sec = ROUND_SECONDS[G.round];
  const el = $(`#take-${side}`);
  m.firstPlayed ??= row;
  if (m.heard[side]) m.replays++;
  $$(".take").forEach((t) => t.classList.remove("playing")); el.classList.add("playing");
  let last = 0;
  const ok = await audio.play(url(G.prompt.id, row), sec, {
    title: `Take ${G.letter(row)}`, sub: `${ROUND_NAMES[G.round]} · ${sec === 30 ? "full 30 s" : `first ${sec} s`}`, key: G.key(row),
    onTick: (e, d) => {
      $(".bar i", el).style.width = `${(100 * e) / d}%`;
      G.listened += Math.max(0, e - last); last = e;
      if (e >= Math.min(3, d) && !m.heard[side] && G.cur === m) { m.heard[side] = true; el.classList.add("heard"); $(".heard-sr", el).textContent = " (heard)"; unlock(); }
    },
  }).catch(() => { $("#hint").textContent = "This clip failed to load. Check your connection and try again."; return false; });
  el.classList.remove("playing");
  return ok;
}

function unlock() {
  const both = G.cur.heard.a && G.cur.heard.b;
  $$(".pick").forEach((p) => (p.disabled = !both)); $("#tie").disabled = !both;
  if (both) $("#hint").innerHTML = 'Both heard. Keep the one you\'d rather keep listening to. <span class="keys">Keys: 1 / 2 keep, 0 coin flip, Q / W replay.</span>';
}

function choose(side) {
  const m = G.cur;
  if (!m || m.decided || !(m.heard.a && m.heard.b)) return;   // one decision per match (key repeat, double click)
  m.decided = true;
  audio.stop();
  const tie = side === "tie";
  if (tie) side = G.rand() < 0.5 ? "a" : "b";
  const winner = side === "a" ? m.left : m.right;
  G.matches.push({ round: G.round, a: m.a, b: m.b, left: m.left, right: m.right, first_played: m.firstPlayed,
                   winner, tie, replays: m.replays, ms: Date.now() - m.t0 });
  $(`#take-${side}`).classList.add("won"); $(`#take-${side === "a" ? "b" : "a"}`).classList.add("lost");
  $$(".pick").forEach((p) => (p.disabled = true)); $("#tie").disabled = true;
  setTimeout(advance, 650);
}

function advance() {
  const nMatches = G.alive.length / 2;
  if (G.match + 1 < nMatches) { G.match++; renderMatch(); return; }
  const winners = G.matches.filter((x) => x.round === G.round).map((x) => x.winner);
  G.stages.push(winners); G.alive = winners; G.match = 0;
  if (G.round === ROUND_SECONDS.length - 1) { finish(); return; }
  G.round++;
  winners.forEach((r) => audio.load(url(G.prompt.id, r)).catch(() => {}));
  const sec = ROUND_SECONDS[G.round];
  $("#inter-eyebrow").textContent = `${ROUND_NAMES[G.round - 1]} done`;
  $("#inter-title").textContent = winners.length > 2 ? `${winners.length} takes advance` : "Two finalists";
  $("#inter-text").textContent = sec === 30
    ? "The final is played in full: 30 seconds each. Everything you cut never had to be finished."
    : `Next round: ${sec} s clips, restarting from the beginning. ${G.prompt.set === "redteam"
      ? "This is a slow-build prompt: the best moments may still be ahead." : "You'll hear where each take is heading."}`;
  $("#next-round").textContent = `Start the ${ROUND_NAMES[G.round].toLowerCase()}`;
  renderBracket($("#inter-bracket"), G.stages, G.seeding, { letter: G.letter, key: G.key });
  show("intermission");
}

// ---------- reveal ----------
function finish() {
  const p = G.prompt, champ = G.stages.at(-1)[0];
  const machineStages = playOut(G.seeding, clapRounds(p)), machine = machineStages.at(-1)[0];
  const ag = agreement(G.matches, p);
  G.champion = champ; G.machine = machine; G.active = false;
  saveGame(ag);

  cover($("#champ-cover"), G.key(champ), G.letter(champ));
  $("#champ-title").textContent = `Take ${G.letter(champ)}`;
  $("#champ-prompt").textContent = `“${p.title}”`;

  $("#agree-num").textContent = `${ag.agree}/${ag.total}`;
  $("#agree-text").textContent = `matches decided the same way as the AI picker (CLAP) would have, on the same pair at the same clip length. `
    + (machine === champ ? `Playing the same draw on its own, it crowned Take ${G.letter(machine)} too.`
                        : `Playing the same draw on its own, it crowned Take ${G.letter(machine)}.`);
  renderBracket($("#machine-bracket"), machineStages, G.seeding, { letter: G.letter, key: G.key });

  const plan = listeningPlan(), h = DATA.headline["schedule_10_0.5"].musiccaps;
  $("#time-num").textContent = `${Math.round(G.listened)} s`;
  $("#time-text").innerHTML = `of listening (the bracket plan is ${plan.restart} s; hearing all eight in full is ${plan.full} s).
    A generator that pruned like this, continuing the survivors instead of restarting, would generate
    <b>${pct(h.compute.mean)}</b> of the audio and still keep its own full-length favourite on
    <b>${pct(h.same_winner.mean)}</b> of prompts <span class="muted">(n = ${h.same_winner.n})</span>.`;

  const outR1 = G.seeding.filter((r) => !G.stages[0].includes(r));
  const lines = Object.entries(CRITICS).map(([j, c]) => {
    const fav = argmax(p.judges[j]);
    return outR1.includes(fav) ? `${c.name}'s favourite (Take ${G.letter(fav)})` : null;
  }).filter(Boolean);
  $("#early-text").innerHTML = lines.length
    ? `In the quarter-finals you knocked out <b>${[...new Set(lines)].join(", ")}</b> after just 10 seconds. `
      + `Early judging can cut a take that only shines later.`
    : `None of the four critics' favourite takes went out in your quarter-finals. Your 10-second ears held up.`;

  const tops = Object.keys(CRITICS).filter((j) => tier(ranks(p.judges[j])[champ]) === "top").length;
  $("#reveal-summary").innerHTML = `You and the AI picker agreed on <b>${ag.agree} of ${ag.total}</b> matches, and
    <b>${tops} of 4</b> critics put your champion in their top two.`;
  $("#breakdown").open = false;
  renderCritics(champ, machine);
  renderTracklist(champ, machine);
  show("reveal");
}

function renderCritics(champ, machine) {
  const el = $("#critics"); el.innerHTML = "";
  for (const [j, c] of Object.entries(CRITICS)) {
    const rk = ranks(G.prompt.judges[j]), fav = argmax(G.prompt.judges[j]);
    const d = document.createElement("article"); d.className = "critic";
    d.innerHTML = `<h3>${c.name}</h3><p class="who">${c.model} (${c.year}): ${c.measures}</p>
      <div class="tier ${tier(rk[champ])}">${TIER_LABEL[tier(rk[champ])]}</div><p class="line">for your champion</p>
      <p class="line">AI picker's champion: <b>${TIER_LABEL[tier(rk[machine])].toLowerCase()}</b> · Their favourite: <b>Take ${G.letter(fav)}</b></p>`;
    el.append(d);
  }
}

function renderTracklist(champ, machine) {
  const el = $("#tracklist"); el.innerHTML = "";
  const outRound = (row) => G.stages.findIndex((s) => !s.includes(row));
  const R = Object.fromEntries(Object.keys(CRITICS).map((j) => [j, ranks(G.prompt.judges[j])]));
  [...G.seeding].sort((x, y) => G.letter(x).localeCompare(G.letter(y))).forEach((row) => {
    const t = document.createElement("div"); t.className = "track";
    const btn = document.createElement("button"); btn.className = "btn round"; btn.textContent = "▶";
    btn.setAttribute("aria-label", `Play take ${G.letter(row)}, full 30 seconds`);
    btn.onclick = () => audio.play(url(G.prompt.id, row), 30, { title: `Take ${G.letter(row)}`, sub: "Full take · 30 s", key: G.key(row) });
    const info = document.createElement("div");
    const o = outRound(row);
    const badges = [row === champ && "your champion", row === machine && "AI picker's champion",
                    o >= 0 && `out in the ${ROUND_NAMES[o].toLowerCase()}`].filter(Boolean).join(" · ");
    info.innerHTML = `<b>Take ${G.letter(row)}</b><div class="badges">${badges}</div>`;
    const rk = document.createElement("div"); rk.className = "ranks";
    for (const [j, c] of Object.entries(CRITICS)) {
      const t = tier(R[j][row]);
      if (t === "middle") continue;
      const s = document.createElement("span"); s.className = `r ${t}`;
      s.textContent = `${c.name.replace("The ", "")} ${t === "top" ? "▲ top two" : "▼ bottom two"}`; rk.append(s);
    }
    t.append(btn, coverEl(G.key(row), G.letter(row), "tiny"), info, rk);
    el.append(t);
  });
}

function saveGame(ag) {
  const games = store.get("eight-takes-games", []);
  games.push({ v: 1, prompt_id: G.prompt.id, seed: G.seed, seeding: G.seeding, matches: G.matches,
               champion: G.stages.at(-1)[0], machine: G.machine, agree: ag.agree, listened_s: Math.round(G.listened),
               when: new Date().toISOString() });
  store.set("eight-takes-games", games.slice(-200));
}

// ---------- how it works ----------
function renderHow() {
  const h = DATA.headline, s = h["schedule_10_0.5"], b = h.bracket, sel = h.selection;
  const ci = (m, f = pct) => `${f(m.lo)}–${f(m.hi)}`;
  const FMT = { pct, int: (x) => String(Math.round(x)), f1: (x) => x.toFixed(1), f2: (x) => sgnless(x, 2) };
  $$("[data-num]").forEach((el) => {
    const v = el.dataset.num.split("/").reduce((o, k) => o?.[k], h);
    el.textContent = typeof v === "number" ? FMT[el.dataset.fmt ?? "f2"](v) : "?";
  });
  const items = [
    [`${pct(s.musiccaps.compute.mean)} of the compute`,
     `Cutting at 10 s and 20 s (8 → 4 → 2) generates ${pct(s.musiccaps.compute.mean)} of the audio, and CLAP still keeps the take it would pick from all eight full songs on ${pct(s.musiccaps.same_winner.mean)} of MusicCaps prompts (95% CI ${ci(s.musiccaps.same_winner)}, n = ${s.musiccaps.same_winner.n}) and ${pct(s.redteam.same_winner.mean)} of slow-build prompts (n = ${s.redteam.same_winner.n}).`,
     s.source],
    [`${pct(b.musiccaps.same_as_unpruned.mean)} survive the draw`,
     `With a random knockout draw like yours (all 315 possible draws per prompt), the AI picker's full-length favourite becomes champion ${pct(b.musiccaps.same_as_unpruned.mean)} of the time (95% CI ${ci(b.musiccaps.same_as_unpruned)}, n = ${b.musiccaps.same_as_unpruned.n}), against ${pct(s.musiccaps.same_winner.mean)} for top-half pruning at the same checkpoints.`,
     b.source],
    ["Is the picker's favourite better?",
     `Barely. Versus a random take, CLAP's pick scores ${sgn(sel.cmi_qual.gain_vs_random.mean)} on CMI-RM quality (95% CI ${sgn(sel.cmi_qual.gain_vs_random.lo)} to ${sgn(sel.cmi_qual.gain_vs_random.hi)}), against a best possible ${sgn(sel.cmi_qual.oracle_gain.mean)}, and ${sgn(sel.mulan.gain_vs_random.mean, 3)} on MuQ-MuLan (CI ${sgn(sel.mulan.gain_vs_random.lo, 3)} to ${sgn(sel.mulan.gain_vs_random.hi, 3)}); n = ${sel.mulan.gain_vs_random.n}. Keeping the same winner cheaply doesn't make the winner good.`,
     sel.source],
  ];
  $("#findings").innerHTML = items.map(([t, body, src]) =>
    `<div class="finding"><b>${t}</b><p>${body}</p><span class="src">Measured · source: ${src}</span></div>`).join("");
}

// ---------- wiring ----------
function wire() {
  const go = (v) => { audio.stop(); if (G) G.active = false; history.replaceState(null, "", v === "how" ? "#how" : "#"); show(v); };
  let pending = null;
  const askLeave = (v) => { pending = v; $("#leave").showModal(); };
  $("#leave-stay").onclick = () => $("#leave").close();
  $("#leave-go").onclick = () => { $("#leave").close(); go(pending); };
  $$("[data-nav]").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault();
    G?.active ? askLeave(a.dataset.nav) : go(a.dataset.nav);
  }));
  window.addEventListener("popstate", () => {          // browser back during a game
    if (G?.active) { history.pushState({ game: true }, "", "#play"); askLeave("home"); }
    else show(location.hash === "#how" ? "how" : "home");
  });
  const keysOn = store.get("eight-takes-keys", true);
  $("#keys-on").checked = keysOn; document.body.classList.toggle("no-keys", !keysOn);
  $("#keys-on").onchange = (e) => { store.set("eight-takes-keys", e.target.checked); document.body.classList.toggle("no-keys", !e.target.checked); };
  $$(".chip").forEach((c) => c.addEventListener("click", () => {
    $$(".chip").forEach((x) => x.classList.toggle("on", x === c)); renderShelf(c.dataset.filter);
  }));
  const pickFrom = (set) => { const ps = DATA.prompts.filter((p) => !set || p.set === set); return ps[Math.floor(Math.random() * ps.length)]; };
  $("#start-random").onclick = () => startGame(pickFrom("musiccaps"));
  $("#start-slow").onclick = () => startGame(pickFrom("redteam"));
  $$(".play").forEach((b) => (b.onclick = () => playSide(b.dataset.side)));
  $$(".pick").forEach((b) => (b.onclick = () => choose(b.dataset.side)));
  $("#tie").onclick = () => choose("tie");
  $("#play-both").onclick = async () => { if (await playSide("a")) await playSide("b"); };
  $("#next-round").onclick = () => { show("game"); renderMatch(); };
  $("#again").onclick = () => { audio.stop(); history.replaceState(null, "", "#"); show("home"); };
  $("#champ-play").onclick = () => audio.play(url(G.prompt.id, G.champion), 30,
    { title: `Take ${G.letter(G.champion)}`, sub: "Your champion · full 30 s", key: G.key(G.champion) });
  $("#np-toggle").onclick = () => { audio.stop(); nowPlaying(); };
  $("#share").onclick = async () => {
    const ag = agreement(G.matches, G.prompt);
    const text = `I crowned Take ${G.letter(G.champion)} in Eight Takes (“${G.prompt.title}”) and agreed with the AI picker on ${ag.agree} of ${ag.total} matches. ${location.href.split("#")[0]}`;
    try { await navigator.clipboard.writeText(text); $("#share").textContent = "Copied!"; }
    catch { $("#share").textContent = "Copy failed"; }
    setTimeout(() => ($("#share").textContent = "Copy result"), 1800);
  };
  document.addEventListener("keydown", (e) => {
    if ($("#game").hidden || !$("#keys-on").checked || $("#leave").open || e.metaKey || e.ctrlKey || e.altKey || e.target.closest("input, textarea")) return;
    const k = e.key.toLowerCase();
    if (k === "q") playSide("a"); else if (k === "w") playSide("b");
    else if (k === "1") choose("a"); else if (k === "2") choose("b"); else if (k === "0") choose("tie");
  });
}

async function main() {
  DATA = await (await fetch("data.json")).json();
  renderShelf(); renderHow(); wire();
  show(location.hash === "#how" ? "how" : "home");
}
main().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<p style="padding:16px">Failed to load: ${e.message}</p>`); });
