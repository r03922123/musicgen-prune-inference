// Lecture page smoke test: renders, every formula and figure renders, no third-party requests,
// no horizontal scroll on a phone, and the link to the game resolves.
//   node lecture/tests/smoke.mjs http://127.0.0.1:8765/        (needs the playwright package)
import { chromium } from "playwright";

const base = process.argv[2] ?? "http://127.0.0.1:8765/";
const browser = await chromium.launch();
const errors = [];
for (const [name, width] of [["desktop", 1280], ["mobile", 390]]) {
  const page = await browser.newPage({ viewport: { width, height: 900 } });
  page.on("pageerror", (e) => errors.push(`${name}: ${e.message}`));
  page.on("console", (m) => m.type() === "error" && errors.push(`${name}: ${m.text()}`));
  page.on("request", (r) => { if (!r.url().startsWith(base)) errors.push(`${name}: third-party request ${r.url()}`); });
  await page.goto(base);
  for (let y = 0; y < 40000; y += 800) {                       // let lazy figures load
    await page.evaluate((yy) => scrollTo(0, yy), y);
    await page.waitForTimeout(30);
  }
  await page.waitForTimeout(500);
  const r = await page.evaluate(() => ({
    h2: document.querySelectorAll("main h2").length,
    toc: document.querySelectorAll(".toc a").length,
    katex: document.querySelectorAll(".katex").length,
    katexErrors: document.querySelectorAll(".katex-error").length,
    rawDollars: [...document.querySelectorAll("main p, main li, main td")].filter((e) => /\$[^$\s][^$]*\$/.test(e.textContent)).length,
    brokenImages: [...document.images].filter((i) => !i.complete || !i.naturalWidth).map((i) => i.src),
    hscroll: document.documentElement.scrollWidth > innerWidth,
    playLink: new URL(document.querySelector('a[href="play/"]').getAttribute("href"), location.href).href,
  }));
  if (r.h2 < 8 || r.toc < r.h2) errors.push(`${name}: sections ${r.h2}, toc entries ${r.toc}`);
  if (!r.katex || r.katexErrors || r.rawDollars) errors.push(`${name}: math katex=${r.katex} errors=${r.katexErrors} unrendered=${r.rawDollars}`);
  if (r.brokenImages.length) errors.push(`${name}: broken images ${r.brokenImages}`);
  if (r.hscroll) errors.push(`${name}: horizontal scroll`);
  if (r.playLink !== new URL("play/", base).href) errors.push(`${name}: play link ${r.playLink}`);
  console.log(`${name}: ${r.h2} sections, ${r.katex} formulas, ${await page.$$eval("figure", (f) => f.length)} figures`);
  await page.close();
}
await browser.close();
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
console.log("lecture ok");
