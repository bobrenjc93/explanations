// shot.js URL OUT.png [SCHEME] [WIDTH] [HEIGHT]: a screenshot of the top of the page.
const puppeteer = require('/home/dev/.npm/_npx/d62b6517736c1e35/node_modules/puppeteer');
(async () => {
  const [url, out, scheme, width, height] = process.argv.slice(2);
  const browser = await puppeteer.launch({executablePath: '/home/dev/.cache/puppeteer/chrome-headless-shell/linux-148.0.7778.97/chrome-headless-shell-linux64/chrome-headless-shell', args: ['--no-sandbox']});
  const page = await browser.newPage();
  await page.setViewport({width: parseInt(width || '1000'), height: parseInt(height || '1500'), deviceScaleFactor: 1});
  if (scheme && scheme !== 'light') await page.emulateMediaFeatures([{name: 'prefers-color-scheme', value: scheme}]);
  await page.goto(url);
  await page.screenshot({path: out});
  console.log('wrote', out);
  await browser.close();
})();
