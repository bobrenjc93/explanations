// shot_full.js URL OUT.png [SCHEME] [WIDTH]: a screenshot of the whole page.
const puppeteer = require('/home/dev/.npm/_npx/d62b6517736c1e35/node_modules/puppeteer');
(async () => {
  const [url, out, scheme, width] = process.argv.slice(2);
  const browser = await puppeteer.launch({executablePath: '/home/dev/.cache/puppeteer/chrome-headless-shell/linux-148.0.7778.97/chrome-headless-shell-linux64/chrome-headless-shell', args: ['--no-sandbox']});
  const page = await browser.newPage();
  await page.setViewport({width: parseInt(width || '1000'), height: 1200, deviceScaleFactor: 1});
  if (scheme && scheme !== 'light') await page.emulateMediaFeatures([{name: 'prefers-color-scheme', value: scheme}]);
  await page.goto(url);
  await page.screenshot({path: out, fullPage: true});
  console.log('wrote', out);
  await browser.close();
})();
