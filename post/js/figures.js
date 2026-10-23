// Interactive figures. Each <figure class="explorable" data-fig="..."> gets mounted here; every
// number they show is recomputed from data.json by sim.js (checked against results/ in CI).
import { agreement, bestOfK, chunkTarget, judgeScores, prune, schedule, spearman } from "./sim.js";

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
let uidN = 0;
const uid = () => `f${++uidN}`;                   // radio-group names, unique per figure instance

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

// Fig: three ways to lay out 4 codebooks per frame. Click a token to see what it can condition on.
function delay(root) {
  const K = 4, S = root.clientWidth < 520 ? 8 : 12;      // fewer steps on a phone, readable cells
  const PATS = { flatten: ["flatten", 4 * 1500, true], parallel: ["parallel", 1500, false], delay: ["delay (MusicGen)", 1503, true] };
  let pat = "delay", sel = { k: 1, f: 3 };
  const ctl = el("div", { class: "controls" }, root);
  radios(ctl, `${uid()}-pat`, "Layout", Object.entries(PATS).map(([k, v]) => [k, v[0]]), pat, (v) => { pat = v; sel = { k: 1, f: pat === "flatten" ? 2 : 3 }; draw(); });
  const lab = el("label", { text: "Step " }, ctl);
  const range = el("input", { type: "range", min: 1, max: S, value: S, "aria-label": "decoding step" }, lab);
  const play = el("button", { type: "button", text: "▶ Play" }, ctl);
  const host = el("div", {}, root), note = el("p", { class: "readout" }, root);
  // the step at which codebook k (0-based) writes frame f (1-based), and its inverse
  const stepOf = (k, f) => (pat === "flatten" ? (f - 1) * K + k + 1 : pat === "parallel" ? f : f + k);
  const frameAt = (k, s) => (pat === "flatten" ? ((s - 1) % K === k ? (s - 1 - k) / K + 1 : null) : pat === "parallel" ? s : s - k);
  const draw = () => {
    const cur = +range.value, sStep = stepOf(sel.k, sel.f);
    const cw = 44, ch = 34, ox = 108, oy = 26, W = ox + S * cw + 8;
    host.replaceChildren();
    const svg = el("svg", { viewBox: `0 0 ${W} ${oy + K * ch + 8}`, role: "img", "aria-label": `${pat} codebook layout` }, host);
    for (let s = 1; s <= S; s++) el("text", { x: ox + (s - 0.5) * cw, y: 16, "text-anchor": "middle", class: "tick", text: s }, svg);
    for (let k = 0; k < K; k++) {
      el("text", { x: ox - 10, y: oy + k * ch + ch / 2 + 4, "text-anchor": "end", class: "tick", text: `codebook ${k + 1}` }, svg);
      for (let s = 1; s <= S; s++) {
        const f = frameAt(k, s);
        if (f === null) continue;                                  // flatten: this codebook is idle at step s
        const on = s <= cur, isSel = f === sel.f && k === sel.k;
        const role = isSel ? " sel" : f >= 1 && s < sStep ? " ctx" : s === sStep ? " blind" : "";
        const g = el("g", { class: f >= 1 ? "pick" : "" }, svg);
        el("rect", { x: ox + (s - 1) * cw + 2, y: oy + k * ch + 2, width: cw - 4, height: ch - 4, rx: 4,
          class: (!on ? "cell off" : f < 1 ? "cell pad" : `cell f${(f - 1) % 6}`) + role }, g);
        el("text", { x: ox + (s - 0.5) * cw, y: oy + k * ch + ch / 2 + 4, "text-anchor": "middle", class: on ? "cell-t" : "cell-t off", text: f >= 1 ? `t${f}` : "pad" }, g);
        if (f >= 1) g.addEventListener("click", () => { sel = { k, f }; draw(); });
      }
    }
    const [name, steps, sees] = PATS[pat];
    const coarser = sel.k > 0 ? stepOf(sel.k - 1, sel.f) < sStep : null;
    note.innerHTML = `<b>${name}</b>: 30 s (1,500 frames) takes <b>${steps.toLocaleString("en")}</b> sequential steps; ` +
      `codebook 2 of a frame ${sees ? "<b>can</b>" : "<b>cannot</b>"} see codebook 1 of the same frame. ` +
      `Selected: codebook ${sel.k + 1} of frame t${sel.f}, written at step ${sStep}. Outlined cells are what it conditions on; ` +
      `dimmed cells are written in the same step, so it cannot see them` +
      (coarser === null ? "." : `, and codebook ${sel.k} of its own frame is ${coarser ? "visible" : "<b>not</b> visible"}.`) + " Click any token.";
  };
  let timer;
  play.addEventListener("click", () => {
    clearInterval(timer); range.value = 1; draw();
    timer = setInterval(() => { if (+range.value >= S) return clearInterval(timer); range.value = +range.value + 1; draw(); }, 450);
  });
  range.addEventListener("input", () => { clearInterval(timer); draw(); });
  draw();
}

// Bootstrap over prompts, seeded, for bands: rows = prompts, cols = values -> {lo, hi} per col.
function bands(rows, nBoot = 1000, seed = 7) {
  let s = seed; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const n = rows.length, m = rows[0].length, means = Array.from({ length: m }, () => []);
  for (let b = 0; b < nBoot; b++) {
    const acc = Array(m).fill(0);
    for (let i = 0; i < n; i++) { const r = rows[Math.floor(rnd() * n)]; for (let j = 0; j < m; j++) acc[j] += r[j]; }
    acc.forEach((v, j) => means[j].push(v / n));
  }
  return means.map((v) => { v.sort((a, b) => a - b); return { lo: v[Math.floor(0.025 * nBoot)], hi: v[Math.ceil(0.975 * nBoot) - 1] }; });
}

// Fig: what N buys, under each judge, picked by the judge itself vs by CLAP, with 95% bands.
// opts.compact: the lead figure (one judge, no controls).
function bestofn(root, data, opts = {}) {
  const ps = data.prompts.filter((p) => p.set === "musiccaps");
  const names = { cmi_qual: "CMI-RM quality", cmi_ali: "CMI-RM alignment", aes_CE: "Audiobox CE", mulan: "MuQ-MuLan" };
  const compact = !!opts.compact;
  let judge = "cmi_qual";
  if (!compact) radios(el("div", { class: "controls" }, root), `${uid()}-judge`, "Grade with", Object.entries(names), judge, (v) => { judge = v; draw(); });
  const host = el("div", {}, root), note = el("p", { class: "readout" }, root);
  const ks = [1, 2, 3, 4, 5, 6, 7, 8];
  const draw = () => {
    const per = (by) => ps.map((p) => ks.map((k) => bestOfK(p, judge, k, by)));
    const mean = (rows) => ks.map((_, j) => rows.reduce((a, r) => a + r[j], 0) / rows.length);
    const oR = per("judge"), cR = per("clap"), oracle = mean(oR), clap = mean(cR), oB = bands(oR), cB = bands(cR);
    const top = Math.max(...oB.map((b) => b.hi)), lo = Math.min(0, ...cB.map((b) => b.lo));
    const step = top > 0.6 ? 0.25 : top > 0.25 ? 0.1 : 0.05;
    const y1 = Math.ceil(top / step) * step, y0 = Math.floor(lo / step) * step;
    const yt = []; for (let y = y0; y <= y1 + 1e-9; y += step) yt.push(+y.toFixed(2));
    host.replaceChildren();
    const W = Math.min(640, Math.max(460, root.clientWidth)), H = compact ? 240 : 300;   // narrower viewBox on phones keeps text legible
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "best-of-N gain chart" }, host);
    const { sx, sy } = axes(svg, { x0: 1, x1: 8, y0, y1, W, H, m: { l: 56, r: 150, t: 28, b: 40 }, xt: ks, yt,
      xl: "N, takes to choose from", yl: `${names[judge]} gain over a random take` });
    el("line", { x1: sx(1), x2: sx(8), y1: sy(0), y2: sy(0), class: "zero" }, svg);
    for (const [vals, b, cls, label] of [[oracle, oB, "s-oracle", "oracle: pick by the judge"], [clap, cB, "s-clap", "pick by CLAP"]]) {
      const band = [...ks.map((k, i) => `${sx(k)},${sy(b[i].hi)}`), ...ks.map((k, i) => `${sx(k)},${sy(b[i].lo)}`).reverse()];
      el("polyline", { points: band.join(" "), class: `band ${cls}` }, svg);
      el("polyline", { points: vals.map((v, i) => `${sx(ks[i])},${sy(v)}`).join(" "), class: `series ${cls}` }, svg);
      vals.forEach((v, i) => el("circle", { cx: sx(ks[i]), cy: sy(v), r: 3.5, class: `dot ${cls}` }, svg));
      el("text", { x: sx(8) + 8, y: sy(vals[7]) + 4, class: `end ${cls}`, text: label }, svg);
    }
    const ry = Math.abs(sy(clap[7]) - sy(0)) < 16 ? sy(clap[7]) + 18 : sy(0) + 14;     // clear of the CLAP label
    el("text", { x: sx(8) + 8, y: ry, class: "end", text: "random pick (= 0)" }, svg);
    // Bands are a visual guide (1,000 in-browser resamples); the intervals quoted in the text come from
    // results/ (10,000 resamples), so the readout does not print the band's own interval.
    note.innerHTML = compact
      ? `Eight takes differ: picking with the judge itself would gain <b>${oracle[7].toFixed(3)}</b>. Picking with CLAP gains <b>${fmt(clap[7], 3)}</b>. MusicCaps, n = ${ps.length}; shaded: 95% bootstrap bands over prompts.`
      : `With 8 takes, a perfect selector gains <b>${oracle[7].toFixed(3)}</b>; CLAP's pick gains <b>${fmt(clap[7], 3)}</b> (interval: the Q2 table). ` +
        `The oracle's first doubling (1→2) buys ${(oracle[1] - oracle[0]).toFixed(3)}, the last (4→8) ${(oracle[7] - oracle[3]).toFixed(3)}. MusicCaps prompts, n = ${ps.length}.`;
  };
  draw();
}

// Fig: successive halving on one prompt, with its audio; and the same schedule over all prompts.
function pruner(root, data, opts = {}) {
  const byId = Object.fromEntries(data.prompts.map((p) => [p.id, p]));
  const st = { id: opts.prompt || data.prompts[0].id, first: 10, keep: 0.5, target: opts.target || "q1" };
  const id = uid();
  const ctl = el("div", { class: "controls" }, root);
  const pl = el("label", { class: "ctl-prompt", text: "Prompt " }, ctl);
  const sel = el("select", { "aria-label": "prompt" }, pl);
  for (const [set, label] of [["redteam", "Slow builds (red team, n = 10)"], ["musiccaps", "MusicCaps (n = 40)"]]) {
    const og = el("optgroup", { label }, sel);
    for (const p of data.prompts.filter((q) => q.set === set)) el("option", { value: p.id, text: p.title }, og);
  }
  sel.value = st.id;
  sel.addEventListener("change", () => { st.id = sel.value; stop(); draw(); });
  radios(ctl, `${id}-first`, "First checkpoint", [[5, "5 s"], [10, "10 s"], [15, "15 s"], [20, "20 s"]], st.first, (v) => { st.first = v; draw(); });
  radios(ctl, `${id}-keep`, "Keep at each", [[0.5, "half"], [0.65, "0.65"], [0.75, "0.75"]], st.keep, (v) => { st.keep = v; draw(); });
  if (opts.target) radios(ctl, `${id}-target`, "Score against", [["q1", "unpruned 30 s CLAP pick"], ["future", "best on unheard 20–30 s"]], st.target, (v) => { st.target = v; draw(); });
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
    // end labels: keep at least 12 px apart so close finishers stay readable
    const labelY = Array(p.n), order = [...Array(p.n).keys()].sort((a, b) => ax.sy(p.clap[30][a]) - ax.sy(p.clap[30][b]));
    order.forEach((i, j) => { const y = ax.sy(p.clap[30][i]); labelY[i] = j ? Math.max(y, labelY[order[j - 1]] + 12) : y; });
    for (let i = 0; i < p.n; i++) {
      const pts = ts.map((t) => [ax.sx(t), ax.sy(p.clap[t][i])]);
      const c = cutAt[i], heard = c === null ? pts : pts.filter((_, j) => ts[j] <= c), rest = c === null ? [] : pts.filter((_, j) => ts[j] >= c);
      const g = el("g", { class: `take k${i}` }, svg);
      if (rest.length) el("polyline", { points: rest.map((q) => q.join(",")).join(" "), class: "ghost" }, g);
      el("polyline", { points: heard.map((q) => q.join(",")).join(" "), class: "live" }, g);
      const end = c === null ? pts[pts.length - 1] : heard[heard.length - 1];
      if (c !== null) el("text", { x: end[0], y: end[1] + 4, "text-anchor": "middle", class: "x", text: "×" }, g);
      const last = pts[pts.length - 1];
      el("text", { x: last[0] + 8, y: labelY[i] + 4, class: "lbl", text: TAKES[i] }, g);
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
      // two decimals, formatted like the post's tables, so figure and table always read the same
      if (st.target === "q1") return `${label} (n = ${s.n}): same winner <b>${s.sameWinner.toFixed(2)}</b>`;
      if (!s.testable) return `${label}: not testable, a checkpoint after 20 s has heard the target`;
      return `${label} (n = ${s.n}): target kept <b>${s.kept.toFixed(2)}</b> vs ${s.chance.toFixed(2)} by chance`;
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

// Fig: do two judges rank the same takes the same way? A bump chart: ranks under CLAP (left) and under
// another judge (right), one line per take; or, zoomed out, one line per prompt (its average take).
function agree(root, data, opts = {}) {
  const names = { clap: "CLAP", mulan: "MuQ-MuLan", cmi_qual: "CMI-RM quality", aes_CE: "Audiobox CE" };
  const st = { other: "mulan", view: "one", id: opts.prompt || data.prompts[0].id };
  const byId = Object.fromEntries(data.prompts.map((p) => [p.id, p])), id = uid();
  const ctl = el("div", { class: "controls" }, root);
  radios(ctl, `${id}-judge`, "Compare CLAP with", Object.entries(names).slice(1), st.other, (v) => { st.other = v; draw(); });
  radios(ctl, `${id}-view`, "Rank", [["one", "the 8 takes of one prompt"], ["all", "the 50 prompts (average take)"]], st.view, (v) => { st.view = v; draw(); });
  const pl = el("label", { class: "ctl-prompt", text: "Prompt " }, ctl);
  const sel = el("select", { "aria-label": "prompt" }, pl);
  for (const p of data.prompts) el("option", { value: p.id, text: p.title }, sel);
  sel.value = st.id;
  sel.addEventListener("change", () => { st.id = sel.value; draw(); });
  const host = el("div", {}, root), note = el("p", { class: "readout" }, root);
  const rankOf = (x) => { const o = [...x.keys()].sort((a, b) => x[b] - x[a] || a - b), r = []; o.forEach((i, j) => (r[i] = j + 1)); return r; };
  const mean = (x) => x.reduce((a, b) => a + b, 0) / x.length;
  const draw = () => {
    pl.style.display = st.view === "one" ? "" : "none";
    const one = st.view === "one", p = byId[st.id];
    const left = one ? judgeScores(p, "clap") : data.prompts.map((q) => mean(judgeScores(q, "clap")));
    const right = one ? judgeScores(p, st.other) : data.prompts.map((q) => mean(judgeScores(q, st.other)));
    const n = left.length, rl = rankOf(left), rr = rankOf(right);
    const W = 420, rowH = one ? 30 : 6.4, top = 44, H = top + n * rowH + 12, xl = 120, xr = W - 120;
    const y = (r) => top + (r - 0.5) * rowH;
    host.replaceChildren();
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "rank agreement between two judges", class: "square" }, host);
    el("text", { x: xl, y: 16, "text-anchor": "middle", class: "label", text: "ranked by CLAP" }, svg);
    el("text", { x: xr, y: 16, "text-anchor": "middle", class: "label", text: `ranked by ${names[st.other]}` }, svg);
    el("text", { x: xl - 30, y: y(1) + 4, "text-anchor": "end", class: "tick", text: "best" }, svg);
    el("text", { x: xl - 30, y: y(n) + 4, "text-anchor": "end", class: "tick", text: "worst" }, svg);
    for (let i = 0; i < n; i++) {
      const g = el("g", { class: one ? `take k${i}` : `rank-p ${data.prompts[i].set}` }, svg);
      el("line", { x1: xl, x2: xr, y1: y(rl[i]), y2: y(rr[i]), class: one ? "live" : "pline" }, g);
      if (one) {
        for (const [x, r, a] of [[xl, rl[i], "end"], [xr, rr[i], "start"]]) {
          el("circle", { cx: x, cy: y(r), r: 5, class: "win" }, g);
          el("text", { x: x + (a === "end" ? -10 : 10), y: y(r) + 4, "text-anchor": a, class: "lbl", text: TAKES[i] }, g);
        }
      }
    }
    const all = agreement(data.prompts, "clap", st.other), other = names[st.other];
    const chance = (1 / 8).toFixed(3);
    if (one) {
      const rho = spearman(left, right), a = rl.indexOf(1), b = rr.indexOf(1);
      note.innerHTML = `This prompt: Spearman ρ = <b>${fmt(rho)}</b>. CLAP's favourite is take <b>${TAKES[a]}</b>, ${other}'s is take <b>${TAKES[b]}</b>` +
        (a === b ? " (the same take)." : ".") +
        ` Over all ${data.prompts.length} prompts: mean within-prompt ρ <b>${fmt(all.within)}</b>; same favourite in <b>${all.topMatch.toFixed(2)}</b> of prompts, against ${chance} for two judges picking at random.`;
    } else {
      note.innerHTML = `Ranking whole prompts by their average take: ρ = <b>${fmt(all.between)}</b> (n = ${n}; blue MusicCaps, orange slow builds). ` +
        `The judges agree more on <i>which prompts</i> came out well than on <i>which take</i> of a prompt is best (ρ ${fmt(all.within)}). Best-of-N only ever needs the second.`;
    }
  };
  draw();
}

const FIGS = { delay, bestofn, pruner, leak, agree };
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
