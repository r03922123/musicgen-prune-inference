// Builds the posts into static pages:  node build.mjs [outDir]   (default: dist/)
//   post.md -> index.html, decoding.md -> decoding/index.html (assets shared at the root)
// Math is rendered here with KaTeX and code highlighted here; KaTeX's CSS/fonts and the highlight
// theme are copied next to the page, so the published page makes no third-party requests.
// Markdown extensions:
//   ^[text]                         a sidenote (margin note on wide screens, tap to open on phones)
//   :::figure name [key=value ...]  an interactive figure (js/figures.js), caption until ":::"
//   In words: ...                   a paragraph restating the formula above it in plain language
import { readFileSync, writeFileSync, mkdirSync, cpSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Marked } from "marked";
import markedKatex from "marked-katex-extension";
import { markedHighlight } from "marked-highlight";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import javascript from "highlight.js/lib/languages/javascript";
import bash from "highlight.js/lib/languages/bash";

hljs.registerLanguage("python", python);
hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("bash", bash);

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(process.argv[2] ?? join(here, "dist"));
const REPO = "https://github.com/r03922123/musicgen-prune-inference";
const HUB = "https://r03922123.github.io/";

const slug = (s) => s.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const esc = (s) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
let toc = [], notes = 0, prefix = "";            // per page; prefix leads from the page to the root

const sidenote = {
  name: "sidenote", level: "inline",
  start: (src) => src.indexOf("^["),
  tokenizer(src) {
    const m = /^\^\[((?:[^\[\]]|\[[^\]]*\])+)\]/.exec(src);
    if (m) return { type: "sidenote", raw: m[0], tokens: this.lexer.inlineTokens(m[1]) };
  },
  renderer(tok) {
    const n = ++notes, id = `sn-${n}`;
    return `<label for="${id}" class="sn-ref" aria-label="note ${n}">${n}</label>` +
      `<input type="checkbox" id="${id}" class="sn-toggle"><span class="sidenote"><b>${n}</b> ${this.parser.parseInline(tok.tokens)}</span>`;
  },
};

const figure = {
  name: "figure", level: "block",
  start: (src) => src.match(/^:::figure /m)?.index,
  tokenizer(src) {
    const m = /^:::figure (\w+)([^\n]*)\n([\s\S]*?)\n:::(?:\n|$)/.exec(src);
    if (!m) return;
    const opts = Object.fromEntries([...m[2].matchAll(/(\w+)=(\S+)/g)].map((x) => [x[1], x[2]]));
    return { type: "figure", raw: m[0], fig: m[1], opts, tokens: this.lexer.inlineTokens(m[3].trim()) };
  },
  renderer(tok) {
    const data = Object.entries(tok.opts).filter(([k]) => k !== "fallback").map(([k, v]) => ` data-${k}="${esc(v)}"`).join("");
    const fb = tok.opts.fallback ? `<img src="${prefix}${tok.opts.fallback}" alt="" loading="lazy">` : "";
    return `<figure class="explorable" data-fig="${tok.fig}"${data}><div class="fig-body">${fb}` +
      `<p class="nojs">This figure is interactive and needs JavaScript.</p></div>` +
      `<figcaption>${this.parser.parseInline(tok.tokens)}</figcaption></figure>\n`;
  },
};

const marked = new Marked(
  markedHighlight({
    langPrefix: "hljs language-",
    highlight: (code, lang) => (hljs.getLanguage(lang) ? hljs.highlight(code, { language: lang }).value : code),
  }),
  markedKatex({ throwOnError: true, nonStandard: true }),
);
marked.use({ extensions: [sidenote, figure] });
marked.use({
  renderer: {
    heading({ tokens, depth }) {
      const text = this.parser.parseInline(tokens);
      if (depth === 1) return `<h1>${text}</h1>\n`;
      const id = slug(text);
      if (depth <= 3) toc.push({ depth, id, text });
      return `<h${depth} id="${id}"><a class="anchor" href="#${id}" aria-hidden="true">#</a>${text}</h${depth}>\n`;
    },
    paragraph({ tokens }) {                       // "In words: ..." restates the formula above it
      const html = this.parser.parseInline(tokens);
      return html.startsWith("In words:") ? `<p class="inwords">${html.slice(9).trim()}</p>\n` : `<p>${html}</p>\n`;
    },
    image({ href, title, text }) {
      const src = prefix + href.replace(/^\.\.\/results\//, "results/");
      const cap = title ? `<figcaption>${marked.parseInline(title)}</figcaption>` : "";
      return `<figure><img src="${src}" alt="${text}" loading="lazy">${cap}</figure>`;
    },
  },
});

function build(file, page_out, pre) {
  toc = []; notes = 0; prefix = pre;
  const md = readFileSync(join(here, file), "utf8");
  // Front matter: "key: value" lines between --- fences at the top.
  const fm = /^---\n([\s\S]*?)\n---\n/.exec(md);
  const meta = Object.fromEntries((fm?.[1] ?? "").split("\n").map((l) => l.split(/:\s(.*)/s).slice(0, 2)));
  const src = fm ? md.slice(fm[0].length) : md;
  const prose = src.replace(/```[\s\S]*?```/g, "").replace(/\$\$[\s\S]*?\$\$/g, "");
  const minutes = Math.round(prose.split(/\s+/).filter(Boolean).length / 220);

  let body = marked.parse(src);
  const tocHtml = `<details class="toc" open><summary>Table of Contents</summary><ul>${toc
    .map((h) => `<li class="d${h.depth}"><a href="#${h.id}">${h.text}</a></li>`).join("")}</ul></details>`;
  body = body.replace("<!-- toc -->", tocHtml);
  const title = (src.match(/^# (.+)$/m) ?? [, "Post"])[1];
  const metaLine = `<p class="post-meta">${meta.date} · ${minutes} min read · ${meta.author}</p>`;
  body = body.replace(/(<h1>[\s\S]*?<\/h1>)/, `$1\n${metaLine}`);

  const page = `<!doctype html>
  <html lang="en">
  <head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${title.replace(/<[^>]+>/g, "")} · r03922123</title>
  <meta name="description" content="${esc(meta.description ?? "")}">
  <link rel="stylesheet" href="${pre}katex/katex.min.css">
  <link rel="stylesheet" href="${pre}hljs/github.min.css" media="(prefers-color-scheme: light)">
  <link rel="stylesheet" href="${pre}hljs/github-dark.min.css" media="(prefers-color-scheme: dark)">
  <link rel="stylesheet" href="${pre}post.css">
  <script type="module" src="${pre}js/figures.js"></script>
  </head>
  <body>
  <header class="site"><div class="wrap">
    <a class="home" href="${HUB}">r03922123</a>
    <nav><a href="${HUB}">Posts</a><a href="${REPO}">Code</a></nav>
  </div></header>
  <main class="wrap">
  <p class="crumbs"><a href="${HUB}">Home</a> » <a href="${HUB}">Posts</a></p>
  ${body}
  </main>
  <footer class="wrap"><p>A personal project, not affiliated with any company. Audio examples are AI-generated with
  MusicGen-small (weights CC-BY-NC 4.0, non-commercial); MusicCaps captions CC BY-SA 4.0. Source of this post:
  <a href="${REPO}/blob/master/post/${file}">post/${file}</a>.</p></footer>
  </body>
  </html>
  `;
  mkdirSync(dirname(join(out, page_out)), { recursive: true });
  writeFileSync(join(out, page_out), page);
  for (const m of md.matchAll(/\.\.\/results\/([\w.-]+\.(?:png|svg))/g)) {
    const f = join(here, "..", "results", m[1]);
    if (!existsSync(f)) throw new Error(`missing figure ${f}`);
    mkdirSync(join(out, "results"), { recursive: true });
    cpSync(f, join(out, "results", m[1]));
}
console.log(`built ${join(out, page_out)}: ${toc.length} headings, ${notes} sidenotes, ` +
  `${(body.match(/class="explorable"/g) ?? []).length} interactive figures, ${minutes} min read`);
}

build("post.md", "index.html", "");
build("decoding.md", "decoding/index.html", "../");

// The old game lived at /play/; send its visitors to the interactive pruning figure.
const redirect = `<!doctype html><meta charset="utf-8"><title>Moved</title>
<meta http-equiv="refresh" content="0; url=../#2-a-cheaper-search-early-pruning"><link rel="canonical" href="../">
<p>The game has been retired. <a href="../#2-a-cheaper-search-early-pruning">Prune the takes yourself in the post</a>.</p>`;

mkdirSync(join(out, "play"), { recursive: true });
writeFileSync(join(out, "play/index.html"), redirect);
cpSync(join(here, "post.css"), join(out, "post.css"));
cpSync(join(here, "figures"), join(out, "figures"), { recursive: true });
cpSync(join(here, "js"), join(out, "js"), { recursive: true });
cpSync(join(here, "data.json"), join(out, "data.json"));
if (existsSync(join(here, "audio"))) cpSync(join(here, "audio"), join(out, "audio"), { recursive: true });
mkdirSync(join(out, "katex"), { recursive: true });
cpSync(join(here, "node_modules/katex/dist/katex.min.css"), join(out, "katex/katex.min.css"));
cpSync(join(here, "node_modules/katex/dist/fonts"), join(out, "katex/fonts"), { recursive: true });
mkdirSync(join(out, "hljs"), { recursive: true });
for (const t of ["github.min.css", "github-dark.min.css"])
  cpSync(join(here, "node_modules/highlight.js/styles", t), join(out, "hljs", t));
