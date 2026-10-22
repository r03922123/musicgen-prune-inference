// Builds lecture.md into a static page:  node build.mjs [outDir]   (default: dist/)
// Math is rendered here with KaTeX, code is highlighted here, and KaTeX's CSS/fonts and the
// highlight theme are copied next to the page, so the published page makes no third-party requests.
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
const md = readFileSync(join(here, "lecture.md"), "utf8");

const slug = (s) => s.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const toc = [];
const marked = new Marked(
  markedHighlight({
    langPrefix: "hljs language-",
    highlight: (code, lang) => (hljs.getLanguage(lang) ? hljs.highlight(code, { language: lang }).value : code),
  }),
  markedKatex({ throwOnError: true, nonStandard: true }),
);
marked.use({
  renderer: {
    heading({ tokens, depth }) {
      const text = this.parser.parseInline(tokens);
      if (depth === 1) return `<h1>${text}</h1>\n`;
      const id = slug(text);
      if (depth <= 3) toc.push({ depth, id, text });
      return `<h${depth} id="${id}"><a class="anchor" href="#${id}" aria-hidden="true">#</a>${text}</h${depth}>\n`;
    },
    image({ href, title, text }) {
      const src = href.replace(/^\.\.\/results\//, "results/");
      const cap = title ? `<figcaption>${marked.parseInline(title)}</figcaption>` : "";
      return `<figure><img src="${src}" alt="${text}" loading="lazy">${cap}</figure>`;
    },
  },
});

let body = marked.parse(md);
const tocHtml = `<nav class="toc" aria-label="Contents"><b>Table of Contents:</b><ul>${toc
  .map((h) => `<li class="d${h.depth}"><a href="#${h.id}">${h.text}</a></li>`).join("")}</ul></nav>`;
body = body.replace("<!-- toc -->", tocHtml);
const title = (md.match(/^# (.+)$/m) ?? [, "Lecture"])[1];

const page = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${title.replace(/<[^>]+>/g, "")}</title>
<meta name="description" content="A graduate-level lecture on best-of-N text-to-music generation: audio tokens, MusicGen, judges and reward models, early pruning, evaluation and system design.">
<link rel="stylesheet" href="katex/katex.min.css">
<link rel="stylesheet" href="hljs/github.min.css" media="(prefers-color-scheme: light)">
<link rel="stylesheet" href="hljs/github-dark.min.css" media="(prefers-color-scheme: dark)">
<link rel="stylesheet" href="lecture.css">
</head>
<body>
<header class="site"><div class="wrap">
  <a class="course" href="./">musicgen-prune-inference</a>
  <nav><a href="./">Lecture</a><a href="play/">Play: Eight Takes</a><a href="https://github.com/r03922123/musicgen-prune-inference">Code</a></nav>
</div></header>
<main class="wrap">
${body}
</main>
<footer class="wrap"><p>A personal project, not affiliated with any company. Audio examples are AI-generated with
MusicGen-small (weights CC-BY-NC 4.0, non-commercial); MusicCaps captions CC BY-SA 4.0. Source of these notes:
<a href="https://github.com/r03922123/musicgen-prune-inference/blob/master/lecture/lecture.md">lecture/lecture.md</a>.</p></footer>
</body>
</html>
`;

mkdirSync(out, { recursive: true });
writeFileSync(join(out, "index.html"), page);
cpSync(join(here, "lecture.css"), join(out, "lecture.css"));
cpSync(join(here, "figures"), join(out, "figures"), { recursive: true });
mkdirSync(join(out, "katex"), { recursive: true });
cpSync(join(here, "node_modules/katex/dist/katex.min.css"), join(out, "katex/katex.min.css"));
cpSync(join(here, "node_modules/katex/dist/fonts"), join(out, "katex/fonts"), { recursive: true });
mkdirSync(join(out, "hljs"), { recursive: true });
for (const t of ["github.min.css", "github-dark.min.css"])
  cpSync(join(here, "node_modules/highlight.js/styles", t), join(out, "hljs", t));
// result figures referenced as ../results/x.png in the markdown
for (const m of md.matchAll(/\.\.\/results\/([\w.-]+\.(?:png|svg))/g)) {
  const src = join(here, "..", "results", m[1]);
  if (!existsSync(src)) throw new Error(`missing figure ${src}`);
  mkdirSync(join(out, "results"), { recursive: true });
  cpSync(src, join(out, "results", m[1]));
}
console.log(`built ${join(out, "index.html")}: ${toc.length} headings, ${(page.length / 1024).toFixed(0)} KB`);
