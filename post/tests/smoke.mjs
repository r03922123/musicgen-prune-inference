// Post page smoke test: renders; every formula, figure and interactive figure renders; the pruning
// figures respond and play audio; no third-party requests; no horizontal scroll on a phone.
//   node post/tests/smoke.mjs http://127.0.0.1:8765/ [screenshotDir]     (needs the playwright package)
import { chromium } from "playwright";

const base = process.argv[2] ?? "http://127.0.0.1:8765/";
const shots = process.argv[3];
const browser = await chromium.launch();
const errors = [];
for (const [name, width] of [["desktop", 1366], ["mobile", 390]]) {
  const page = await browser.newPage({ viewport: { width, height: 900 } });
  page.on("pageerror", (e) => errors.push(`${name}: ${e.message}`));
  page.on("console", (m) => m.type() === "error" && errors.push(`${name}: ${m.text()}`));
  page.on("request", (r) => { if (!r.url().startsWith(base)) errors.push(`${name}: third-party request ${r.url()}`); });
  await page.goto(base);
  await page.waitForSelector("figure.explorable.live", { timeout: 10000 });
  for (let y = 0; y < 60000; y += 800) {                       // let lazy figures load
    await page.evaluate((yy) => scrollTo(0, yy), y);
    await page.waitForTimeout(25);
  }
  await page.waitForTimeout(400);
  const r = await page.evaluate(() => ({
    h2: document.querySelectorAll("main h2").length,
    toc: document.querySelectorAll(".toc a").length,
    katex: document.querySelectorAll(".katex").length,
    katexErrors: document.querySelectorAll(".katex-error").length,
    rawDollars: [...document.querySelectorAll("main p, main li, main td")].filter((e) => /\$[^$\s][^$]*\$/.test(e.textContent)).length,
    brokenImages: [...document.images].filter((i) => !i.complete || !i.naturalWidth).map((i) => i.src),
    hscroll: document.documentElement.scrollWidth > innerWidth,
    live: [...document.querySelectorAll("figure.explorable")].map((f) => [f.dataset.fig, f.classList.contains("live") && !!f.querySelector("svg")]),
    sidenotes: document.querySelectorAll(".sidenote").length,
  }));
  if (r.h2 < 8 || r.toc < r.h2) errors.push(`${name}: sections ${r.h2}, toc entries ${r.toc}`);
  if (!r.katex || r.katexErrors || r.rawDollars) errors.push(`${name}: math katex=${r.katex} errors=${r.katexErrors} unrendered=${r.rawDollars}`);
  if (r.brokenImages.length) errors.push(`${name}: broken images ${r.brokenImages}`);
  if (r.hscroll) errors.push(`${name}: horizontal scroll`);
  if (r.live.length !== 6 || r.live.some(([, ok]) => !ok)) errors.push(`${name}: interactive figures ${JSON.stringify(r.live)}`);
  if (r.sidenotes < 3) errors.push(`${name}: sidenotes ${r.sidenotes}`);

  // The pruning figure responds to its controls and recomputes over all prompts.
  const fig = page.locator('figure[data-fig="pruner"]').first();
  const before = await fig.locator(".readout").last().textContent();
  await fig.locator('input[name$="-first"][value="5"]').check();
  const after = await fig.locator(".readout").last().textContent();
  if (before === after || !/same winner 0\.30/.test(after)) errors.push(`${name}: pruner readout did not update: ${after}`);
  if (await fig.locator('input[name$="-target"]').count()) errors.push(`${name}: Fig. 5 should not offer the target toggle`);
  // Fig. 7: the leak-free target, and its testability rule.
  const fig7 = page.locator('figure[data-fig="pruner"]').nth(1);
  if (!/target kept 0\.72 vs 0\.25 by chance/.test(await fig7.locator(".readout").last().textContent())) errors.push(`${name}: Fig. 7 kept rate`);
  await fig7.locator('input[name$="-first"][value="15"]').check();
  if (!/kept/.test(await fig7.locator(".readout").last().textContent())) errors.push(`${name}: Fig. 7 at 15 s should be testable`);
  // Fig. 1: switching layout changes the step count.
  const d = page.locator('figure[data-fig="delay"]');
  await d.locator('input[value="flatten"]').check();
  if (!/6,000/.test(await d.locator(".readout").textContent())) errors.push(`${name}: delay layout readout`);
  // Audio: pressing a take requests its MP3 and starts playing.
  const audioReq = page.waitForResponse((res) => /audio\/rt_00_3\.mp3$/.test(res.url()), { timeout: 8000 }).catch(() => null);
  await fig.locator(".take-b").nth(3).click();
  const res = await audioReq;
  if (!res || res.status() >= 400) errors.push(`${name}: take audio ${res ? res.status() : "not requested"}`);
  await fig.locator(".take-b").nth(3).click();                    // stop
  if (shots) {
    await fig.scrollIntoViewIfNeeded();
    await fig.screenshot({ path: `${shots}/pruner-${name}.png` });
    await page.screenshot({ path: `${shots}/top-${name}.png` });
    for (const [i, f] of [[0, "delay"], [1, "bestofn"], [0, "leak"], [1, "pruner"]])
      await page.locator(`figure[data-fig="${f}"]`).nth(i).screenshot({ path: `${shots}/${f}${i}-${name}.png` });
  }
  console.log(`${name}: ${r.h2} sections, ${r.katex} formulas, ${r.live.length} interactive figures, ${r.sidenotes} sidenotes`);
  await page.close();
}
// The retired game's address redirects into the post.
const p = await browser.newPage();
await p.goto(new URL("play/", base).href);
await p.waitForURL(/#early-pruning$/, { timeout: 5000 }).catch(() => errors.push("play/ does not redirect to the post"));
await browser.close();
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
console.log("post ok");
