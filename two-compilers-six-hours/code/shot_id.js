// shot_id.js URL OUT.png ID HEIGHT [SCHEME]: a screenshot of the page from the element with this id down.
const puppeteer = require('/home/dev/.npm/_npx/d62b6517736c1e35/node_modules/puppeteer');
(async () => {
  const [url, out, id, height, scheme] = process.argv.slice(2);
  const browser = await puppeteer.launch({executablePath: '/home/dev/.cache/puppeteer/chrome-headless-shell/linux-148.0.7778.97/chrome-headless-shell-linux64/chrome-headless-shell', args: ['--no-sandbox']});
  const page = await browser.newPage();
  await page.setViewport({width: 1000, height: 900, deviceScaleFactor: 1});
  if (scheme && scheme !== 'light') await page.emulateMediaFeatures([{name: 'prefers-color-scheme', value: scheme}]);
  await page.goto(url);
  const box = await page.evaluate((i) => { const r = document.getElementById(i).getBoundingClientRect(); return {y: r.top + window.scrollY, w: document.documentElement.clientWidth}; }, id);
  await page.screenshot({path: out, clip: {x: 0, y: Math.max(0, box.y - 10), width: box.w, height: parseInt(height)}});
  console.log('wrote', out);
  await browser.close();
})();
