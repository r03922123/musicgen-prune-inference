// Interactive figures. Each <figure class="explorable" data-fig="..."> gets mounted here; every
// number they show is recomputed from data.json by sim.js (checked against results/ in CI).
import { prune, schedule } from "./sim.js";

const NS = "http://www.w3.org/2000/svg";
const TAKES = "ABCDEFGH";
const el = (tag, attrs = {}, parent) => {
  const e = tag === "svg" || ["g", "line", "path", "circle", "rect", "text", "polyline"].includes(tag)
    ? document.createElementNS(NS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === "text" ? (e.textContent = v) : e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
};
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

// Fig: successive halving on one prompt, with its audio; and the same schedule over all prompts.
function pruner(root, data, opts = {}) {
  const byId = Object.fromEntries(data.prompts.map((p) => [p.id, p]));
  const st = { id: opts.prompt || data.prompts[0].id, first: 10, keep: 0.5 };
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
    const goal = r.base;
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
        audio.src = new URL(`../audio/${p.id}_${i}.mp3`, import.meta.url).href; audio.play().catch(() => {});
        head.style.display = ""; tick();
      });
    }
    const goalName = "the take unpruned best-of-8 picks";
    mine.innerHTML = `This prompt: ${r.survivors.length} of 8 takes reach 30 s, compute <b>${r.compute.toFixed(3)}</b> of full best-of-8. ` +
      (r.survivors.includes(goal) ? `◎ ${goalName} (${TAKES[goal]}) <b>survives</b>.` : `◎ ${goalName} (${TAKES[goal]}) is <b>cut at ${cutAt[goal]} s</b>.`);
    const inResults = [5, 10].includes(st.first) ? "" : " (not in the committed results: explore freely)";
    const line = (set, label) => {
      const s = schedule(data.prompts.filter((q) => q.set === set), st.first, st.keep);
      // two decimals, formatted like the post's tables, so figure and table always read the same
      return `${label} (n = ${s.n}): same winner <b>${s.sameWinner.toFixed(2)}</b>`;
    };
    all.innerHTML = `All prompts, this schedule${inResults}: ${line("musiccaps", "MusicCaps")}; ${line("redteam", "slow builds")}.`;
  };
  draw();
}

const FIGS = { delay, pruner };
const nodes = [...document.querySelectorAll("figure.explorable")];
if (nodes.length) {
  fetch(new URL("../data.json", import.meta.url)).then((r) => r.json()).then((data) => {
    for (const f of nodes) {
      const body = f.querySelector(".fig-body");
      body.replaceChildren();
      FIGS[f.dataset.fig](body, data, f.dataset);
      f.classList.add("live");
    }
  });
}
