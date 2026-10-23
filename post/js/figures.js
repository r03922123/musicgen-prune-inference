// Interactive figures. Each <figure class="explorable" data-fig="..."> gets mounted here; every
// number they show is recomputed from data.json by sim.js (checked against results/ in CI).
import { bestOfK, checkpoints, chunkTarget, prune, schedule, spearman } from "./sim.js";

const NS = "http://www.w3.org/2000/svg";
const TAKES = "ABCDEFGH";
const el = (tag, attrs = {}, parent) => {
  const e = tag === "svg" || ["g", "line", "path", "circle", "rect", "text", "polyline"].includes(tag)
    ? document.createElementNS(NS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === "text" ? (e.textContent = v) : e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
};
const fmt = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const pct = (x) => `${Math.round(100 * x)}%`;

function radios(parent, name, label, options, value, onChange) {
  const box = el("fieldset", { class: "ctl" }, parent);
  el("legend", { text: label }, box);
  for (const [v, text] of options) {
    const l = el("label", {}, box);
    const i = el("input", { type: "radio", name, value: String(v) }, l);
    if (String(v) === String(value)) i.checked = true;
    i.addEventListener("change", () => onChange(v));
    l.append(" " + text);
  }
  return box;
}

function axes(svg, { x0, x1, y0, y1, W, H, m, xt, yt, xl, yl }) {
  const sx = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
  const sy = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
  const g = el("g", { class: "axis" }, svg);
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, g);
  el("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b }, g);
  for (const t of xt) el("text", { x: sx(t), y: H - m.b + 16, "text-anchor": "middle", text: t }, g);
  for (const t of yt) {
    el("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: "grid" }, g);
    el("text", { x: m.l - 6, y: sy(t) + 4, "text-anchor": "end", text: t }, g);
  }
  if (xl) el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "label", text: xl }, g);
  if (yl) el("text", { x: 4, y: 12, class: "label", text: yl }, g);
  return { sx, sy };
}

// Fig: the delay pattern, one decoding step at a time.
function delay(root) {
  const K = 4, S = root.clientWidth < 520 ? 7 : 12;      // fewer steps on a phone, readable cells
  const ctl = el("div", { class: "controls" }, root);
  const lab = el("label", { text: "Decoding step " }, ctl);
  const range = el("input", { type: "range", min: 1, max: S, value: 5, "aria-label": "decoding step" }, lab);
  const out = el("output", {}, lab);
  const play = el("button", { type: "button", text: "▶ Play" }, ctl);
  const cw = 44, ch = 34, ox = 108, oy = 26, W = ox + S * cw + 8;
  const svg = el("svg", { viewBox: `0 0 ${W} ${oy + K * ch + 30}`, role: "img", "aria-label": "delay pattern grid" }, root);
  for (let s = 1; s <= S; s++) el("text", { x: ox + (s - 0.5) * cw, y: 16, "text-anchor": "middle", class: "tick", text: s }, svg);
  const cells = [];
  for (let k = 0; k < K; k++) {
    el("text", { x: ox - 10, y: oy + k * ch + ch / 2 + 4, "text-anchor": "end", class: "tick", text: `codebook ${k + 1}` }, svg);
    for (let s = 1; s <= S; s++) {
      const f = s - k, g = el("g", {}, svg);
      const r = el("rect", { x: ox + (s - 1) * cw + 2, y: oy + k * ch + 2, width: cw - 4, height: ch - 4, rx: 4 }, g);
      const t = el("text", { x: ox + (s - 0.5) * cw, y: oy + k * ch + ch / 2 + 4, "text-anchor": "middle", text: f >= 1 ? `t${f}` : "pad" }, g);
      cells.push({ s, f, r, t });
    }
  }
  const cursor = el("rect", { y: oy - 2, width: cw, height: K * ch + 4, class: "cursor", rx: 5 }, svg);
  const note = el("p", { class: "readout" }, root);
  const draw = () => {
    const s = +range.value;
    out.textContent = ` ${s}`;
    cursor.setAttribute("x", ox + (s - 1) * cw);
    for (const c of cells) {
      const on = c.s <= s;
      c.r.setAttribute("class", !on ? "cell off" : c.f < 1 ? "cell pad" : `cell f${(c.f - 1) % 6}`);
      c.t.setAttribute("class", on ? "cell-t" : "cell-t off");
    }
    const done = s - (K - 1);
    note.innerHTML = `Step ${s} writes ` + [...Array(K).keys()].map((k) => s - k >= 1
      ? `frame <b>t${s - k}</b> of codebook ${k + 1}` : `padding for codebook ${k + 1}`).join(", ") +
      `. ${done >= 1 ? `Frames t1–t${done} now have all four codebooks.` : "No frame is complete yet."}` +
      " Codebook 2 never writes a frame before codebook 1 has.";
  };
  let timer;
  play.addEventListener("click", () => {
    clearInterval(timer);
    range.value = 1; draw();
    timer = setInterval(() => { if (+range.value >= S) return clearInterval(timer); range.value = +range.value + 1; draw(); }, 550);
  });
  range.addEventListener("input", () => { clearInterval(timer); draw(); });
  draw();
}

// Fig: what N buys, under each judge, picked by the judge itself vs by CLAP.
function bestofn(root, data) {
  const ps = data.prompts.filter((p) => p.set === "musiccaps");
  const names = { cmi_qual: "CMI-RM quality", cmi_ali: "CMI-RM alignment", aes_CE: "Audiobox CE", mulan: "MuQ-MuLan" };
  let judge = "cmi_qual";
  const ctl = el("div", { class: "controls" }, root);
  radios(ctl, "bon-judge", "Grade with", Object.entries(names), judge, (v) => { judge = v; draw(); });
  const host = el("div", {}, root), note = el("p", { class: "readout" }, root);
  const draw = () => {
    const ks = [1, 2, 3, 4, 5, 6, 7, 8];
    const avg = (k, by) => ps.reduce((a, p) => a + bestOfK(p, judge, k, by), 0) / ps.length;
    const oracle = ks.map((k) => avg(k, "judge")), clap = ks.map((k) => avg(k, "clap"));
    const top = Math.max(...oracle), lo = Math.min(0, ...clap);
    const step = top > 0.6 ? 0.25 : top > 0.25 ? 0.1 : 0.05;
    const y1 = Math.ceil(top / step) * step, y0 = Math.floor(lo / step) * step;
    const yt = []; for (let y = y0; y <= y1 + 1e-9; y += step) yt.push(+y.toFixed(2));
    host.replaceChildren();
    const W = Math.min(640, Math.max(460, root.clientWidth)), H = 300;   // narrower viewBox on phones keeps text legible
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "best-of-N gain chart" }, host);
    const { sx, sy } = axes(svg, { x0: 1, x1: 8, y0, y1, W, H, m: { l: 56, r: 150, t: 28, b: 40 }, xt: ks, yt,
      xl: "N, takes to choose from", yl: `${names[judge]} gain over a random take` });
    el("line", { x1: sx(1), x2: sx(8), y1: sy(0), y2: sy(0), class: "zero" }, svg);
    for (const [vals, cls, label] of [[oracle, "s-oracle", "oracle: pick by the judge"], [clap, "s-clap", "pick by CLAP"]]) {
      el("polyline", { points: vals.map((v, i) => `${sx(ks[i])},${sy(v)}`).join(" "), class: `series ${cls}` }, svg);
      vals.forEach((v, i) => el("circle", { cx: sx(ks[i]), cy: sy(v), r: 3.5, class: `dot ${cls}` }, svg));
      el("text", { x: sx(8) + 8, y: sy(vals[7]) + 4, class: `end ${cls}`, text: label }, svg);
    }
    el("text", { x: sx(8) + 8, y: sy(0) + 14, class: "end", text: "random pick (= 0)" }, svg);
    note.innerHTML = `With 8 takes, the best a perfect selector could gain is <b>${oracle[7].toFixed(3)}</b>; CLAP's pick ` +
      `gains <b>${fmt(clap[7], 3)}</b>. The oracle's first doubling (1→2) buys ${(oracle[1] - oracle[0]).toFixed(3)}, ` +
      `the last (4→8) ${(oracle[7] - oracle[3]).toFixed(3)}. MusicCaps prompts, n = ${ps.length}, averaged over every subset of N takes.`;
  };
  draw();
}

// Fig: successive halving on one prompt, with its audio; and the same schedule over all prompts.
function pruner(root, data, opts = {}) {
  const byId = Object.fromEntries(data.prompts.map((p) => [p.id, p]));
  const st = { id: opts.prompt || data.prompts[0].id, first: 10, keep: 0.5, target: "q1" };
  const ctl = el("div", { class: "controls" }, root);
  const pl = el("label", { class: "ctl-prompt", text: "Prompt " }, ctl);
  const sel = el("select", { "aria-label": "prompt" }, pl);
  for (const [set, label] of [["redteam", "Slow builds (red team, n = 10)"], ["musiccaps", "MusicCaps (n = 40)"]]) {
    const og = el("optgroup", { label }, sel);
    for (const p of data.prompts.filter((q) => q.set === set)) el("option", { value: p.id, text: p.title }, og);
  }
  sel.value = st.id;
  sel.addEventListener("change", () => { st.id = sel.value; stop(); draw(); });
  radios(ctl, "pr-first", "First checkpoint", [[5, "5 s"], [10, "10 s"], [15, "15 s"], [20, "20 s"]], st.first, (v) => { st.first = v; draw(); });
  radios(ctl, "pr-keep", "Keep at each", [[0.5, "half"], [0.65, "0.65"], [0.75, "0.75"]], st.keep, (v) => { st.keep = v; draw(); });
  radios(ctl, "pr-target", "Score against", [["q1", "unpruned 30 s CLAP pick"], ["future", "best on unheard 20–30 s"]], st.target, (v) => { st.target = v; draw(); });
  const caption = el("p", { class: "prompt-caption" }, root);
  const host = el("div", {}, root);
  const takes = el("div", { class: "takes", role: "group", "aria-label": "play takes" }, root);
  const mine = el("p", { class: "readout" }, root), all = el("p", { class: "readout" }, root);
  const audio = new Audio();
  let playing = null, head = null, raf = 0, sx = null;
  const stop = () => { audio.pause(); playing = null; cancelAnimationFrame(raf); if (head) head.style.display = "none"; for (const b of takes.children) b.classList.remove("on"); };
  audio.addEventListener("ended", stop);
  audio.addEventListener("error", () => { if (playing !== null) { stop(); all.insertAdjacentHTML("afterbegin", "<b>Audio is not available in this build.</b> "); } });
  const tick = () => { if (head && sx) { head.setAttribute("x1", sx(audio.currentTime)); head.setAttribute("x2", sx(audio.currentTime)); } raf = requestAnimationFrame(tick); };

  const draw = () => {
    const p = byId[st.id], r = prune(p, st.first, st.keep);
    const goal = st.target === "q1" ? r.base : r.target;
    caption.textContent = `“${p.caption}”`;
    const cutAt = Array(p.n).fill(null);
    for (const [t, alive] of Object.entries(r.alive)) for (let i = 0; i < p.n; i++) if (cutAt[i] === null && !alive.includes(i)) cutAt[i] = +t;
    const ts = data.prefixes, vals = ts.flatMap((t) => p.clap[t]);
    const y0 = Math.floor(Math.min(...vals) * 20) / 20, y1 = Math.ceil(Math.max(...vals) * 20) / 20;
    const yt = []; for (let y = y0; y <= y1 + 1e-9; y += 0.05) yt.push(+y.toFixed(2));
    host.replaceChildren();
    const W = Math.min(640, Math.max(400, root.clientWidth)), H = 320;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "CLAP score of each take's prefix over time, with pruning" }, host);
    const ax = axes(svg, { x0: 0, x1: 30, y0, y1, W, H, m: { l: 52, r: 40, t: 28, b: 40 }, xt: [0, 5, 10, 15, 20, 25, 30], yt,
      xl: "seconds generated", yl: "CLAP score of the prefix so far" });
    sx = ax.sx;
    for (const t of r.cps.filter((t) => t in r.alive)) {
      el("line", { x1: ax.sx(t), x2: ax.sx(t), y1: 28, y2: H - 40, class: "cp" }, svg);
      el("text", { x: ax.sx(t) + 4, y: 42, class: "cp-t", text: `keep ${r.alive[t].length}` }, svg);
    }
    if (st.target === "future") el("rect", { x: ax.sx(20), y: 28, width: ax.sx(30) - ax.sx(20), height: H - 68, class: "unheard" }, svg);
    for (let i = 0; i < p.n; i++) {
      const pts = ts.map((t) => [ax.sx(t), ax.sy(p.clap[t][i])]);
      const c = cutAt[i], heard = c === null ? pts : pts.filter((_, j) => ts[j] <= c), rest = c === null ? [] : pts.filter((_, j) => ts[j] >= c);
      const g = el("g", { class: `take k${i}` }, svg);
      if (rest.length) el("polyline", { points: rest.map((q) => q.join(",")).join(" "), class: "ghost" }, g);
      el("polyline", { points: heard.map((q) => q.join(",")).join(" "), class: "live" }, g);
      const end = c === null ? pts[pts.length - 1] : heard[heard.length - 1];
      if (c !== null) el("text", { x: end[0], y: end[1] + 4, "text-anchor": "middle", class: "x", text: "×" }, g);
      const last = pts[pts.length - 1];
      el("text", { x: last[0] + 8, y: last[1] + 4, class: "lbl", text: TAKES[i] }, g);
      if (i === r.winner) el("circle", { cx: last[0], cy: last[1], r: 5, class: "win" }, g);
      if (i === goal) el("circle", { cx: last[0], cy: last[1], r: 9, class: "goal" }, g);
    }
    head = el("line", { y1: 28, y2: H - 40, class: "playhead" }, svg);
    head.style.display = playing === null ? "none" : "";
    takes.replaceChildren();
    for (let i = 0; i < p.n; i++) {
      const fate = i === r.winner ? "picked" : cutAt[i] === null ? "finalist" : `cut at ${cutAt[i]} s`;
      const b = el("button", { type: "button", class: `take-b k${i}${playing === i ? " on" : ""}`, title: `Play take ${TAKES[i]} (${fate})` }, takes);
      b.innerHTML = `<span class="sw"></span>▶ ${TAKES[i]} <small>${fate}${i === goal ? " ◎" : ""}</small>`;
      b.addEventListener("click", () => {
        if (playing === i) return stop();
        stop(); playing = i; b.classList.add("on");
        audio.src = `audio/${p.id}_${i}.mp3`; audio.play().catch(() => {});
        head.style.display = ""; tick();
      });
    }
    const goalName = st.target === "q1" ? "the take unpruned best-of-8 picks" : "the take CLAP likes best on the last 10 s";
    mine.innerHTML = `This prompt: ${r.survivors.length} of 8 takes reach 30 s, compute <b>${r.compute.toFixed(3)}</b> of full best-of-8. ` +
      (r.survivors.includes(goal) ? `◎ ${goalName} (${TAKES[goal]}) <b>survives</b>.` : `◎ ${goalName} (${TAKES[goal]}) is <b>cut at ${cutAt[goal]} s</b>.`);
    const inResults = [5, 10].includes(st.first) ? "" : " (not in the committed results: explore freely)";
    const line = (set, label) => {
      const s = schedule(data.prompts.filter((q) => q.set === set), st.first, st.keep);
      if (st.target === "q1") return `${label} (n = ${s.n}): same winner <b>${pct(s.sameWinner)}</b>`;
      if (!s.testable) return `${label}: not testable, a checkpoint after 20 s has heard the target`;
      return `${label} (n = ${s.n}): target kept <b>${pct(s.kept)}</b> vs ${pct(s.chance)} by chance`;
    };
    all.innerHTML = `All prompts, this schedule${inResults}: ${line("musiccaps", "MusicCaps")}; ${line("redteam", "slow builds")}.`;
  };
  draw();
}

// Fig: the 10 s score against the chunk it has heard vs the chunks it has not.
function leak(root, data) {
  let which = "heard";
  const ctl = el("div", { class: "controls" }, root);
  radios(ctl, "lk", "Compare the 10 s score with", [["heard", "CLAP on 0–10 s (heard)"], ["unheard", "CLAP on 10–30 s (unheard)"]], which, (v) => { which = v; draw(); });
  const host = el("div", {}, root), note = el("p", { class: "readout" }, root);
  const ranks = (x) => x.map((v) => 1 + x.filter((w) => w < v).length + (x.filter((w) => w === v).length - 1) / 2);
  const draw = () => {
    host.replaceChildren();
    const W = 420, H = 380, svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "rank scatter", class: "square" }, host);
    const ax = axes(svg, { x0: 0.5, x1: 8.5, y0: 0.5, y1: 8.5, W, H, m: { l: 44, r: 16, t: 28, b: 40 }, xt: [1, 2, 3, 4, 5, 6, 7, 8], yt: [1, 2, 3, 4, 5, 6, 7, 8],
      xl: "rank of a take by its 10 s score (within its prompt)", yl: "rank by the target" });
    const rhos = { musiccaps: [], redteam: [] };
    let seed = 1; const jit = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.55;
    for (const p of data.prompts) {
      const x = ranks(p.clap["10"]), y = ranks(chunkTarget(p, which));
      const rho = spearman(p.clap["10"], chunkTarget(p, which));
      if (!Number.isNaN(rho)) rhos[p.set].push(rho);
      x.forEach((xv, i) => el("circle", { cx: ax.sx(xv + jit()), cy: ax.sy(y[i] + jit()), r: 2.6, class: `pt ${p.set}` }, svg));
    }
    const m = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    note.innerHTML = `Within-prompt Spearman ρ, averaged: MusicCaps <b>${fmt(m(rhos.musiccaps))}</b> (n = ${rhos.musiccaps.length}), ` +
      `slow builds <b>${fmt(m(rhos.redteam))}</b> (n = ${rhos.redteam.length}). ` +
      (which === "heard" ? "The 10 s score is, up to decoding noise, the same measurement as the first chunk."
        : "This is what an early score can actually predict about audio it has not heard.");
  };
  draw();
}

const FIGS = { delay, bestofn, pruner, leak };
const nodes = [...document.querySelectorAll("figure.explorable")];
if (nodes.length) {
  fetch("data.json").then((r) => r.json()).then((data) => {
    for (const f of nodes) {
      const body = f.querySelector(".fig-body");
      body.replaceChildren();
      FIGS[f.dataset.fig](body, data, f.dataset);
      f.classList.add("live");
    }
  });
}
