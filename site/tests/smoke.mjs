// End-to-end smoke test: plays one full game in headless Chromium and screenshots each view.
//   python3 -m http.server 8765 --directory site &   then
//   node site/tests/smoke.mjs [base-url] [screenshot-dir]      (needs the playwright package)
import { chromium } from "playwright";

const base = process.argv[2] ?? "http://127.0.0.1:8765/";
const shots = process.argv[3] ?? ".";
const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
const errors = [];
for (const [name, viewport] of [["desktop", { width: 1280, height: 860 }], ["mobile", { width: 390, height: 844 }]]) {
  const page = await browser.newPage({ viewport });
  page.on("console", (m) => m.type() === "error" && errors.push(`${name}: ${m.text()}`));
  page.on("pageerror", (e) => errors.push(`${name}: ${e.message}`));
  await page.goto(base);
  await page.waitForSelector(".tile");
  await page.waitForTimeout(400);                                 // let the view fade in
  if ((await page.$$(".tile")).length !== 50) throw new Error("expected 50 prompt tiles");
  await page.screenshot({ path: `${shots}/home-${name}.png`, fullPage: false });
  await page.click("#start-slow");
  for (let m = 0; m < 7; m++) {
    await page.click('.play[data-side="a"]');
    await page.waitForSelector("#take-a.heard", { timeout: 15000 });
    await page.click('.play[data-side="b"]');
    await page.waitForSelector("#take-b.heard", { timeout: 15000 });
    if (m === 0) await page.screenshot({ path: `${shots}/match-${name}.png` });
    await page.click(`.pick[data-side="${m % 2 ? "b" : "a"}"]`);
    await page.waitForTimeout(800);
    if (await page.isVisible("#intermission")) {
      if (m === 3) await page.screenshot({ path: `${shots}/intermission-${name}.png` });
      await page.click("#next-round");
    }
  }
  await page.waitForSelector("#reveal:not([hidden])");
  const agree = await page.textContent("#agree-num");
  if (!/^\d\/7$/.test(agree)) throw new Error(`agreement shows ${agree}`);
  if ((await page.$$(".critic")).length !== 4 || (await page.$$(".track")).length !== 8) throw new Error("reveal incomplete");
  await page.screenshot({ path: `${shots}/reveal-${name}.png`, fullPage: true });
  await page.click('[data-nav="how"]');
  await page.waitForTimeout(400);
  const unfilled = await page.$$eval("[data-num]", (els) => els.filter((e) => !e.textContent || e.textContent === "?").length);
  if (unfilled) throw new Error(`${unfilled} article numbers not filled from data.json`);
  await page.screenshot({ path: `${shots}/how-${name}.png`, fullPage: true });
  const games = await page.evaluate(() => JSON.parse(localStorage.getItem("eight-takes-games")));
  if (games.length !== 1 || games[0].matches.length !== 7) throw new Error("game log not saved");
  console.log(`${name}: game complete, agreement ${agree}`);
  await page.close();
}
await browser.close();
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
console.log("no console errors");
