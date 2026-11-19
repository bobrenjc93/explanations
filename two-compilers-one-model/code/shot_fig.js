// shot_fig.js URL OUT.png INDEX [SCHEME]: a screenshot of the INDEX-th figure of the page, at twice the size.
const puppeteer = require('/home/dev/.npm/_npx/d62b6517736c1e35/node_modules/puppeteer');
(async () => {
  const [url, out, index, scheme] = process.argv.slice(2);
  const browser = await puppeteer.launch({executablePath: '/home/dev/.cache/puppeteer/chrome-headless-shell/linux-148.0.7778.97/chrome-headless-shell-linux64/chrome-headless-shell', args: ['--no-sandbox']});
  const page = await browser.newPage();
  await page.setViewport({width: 1000, height: 1200, deviceScaleFactor: 2});
  if (scheme && scheme !== 'light') await page.emulateMediaFeatures([{name: 'prefers-color-scheme', value: scheme}]);
  await page.goto(url);
  const figures = await page.$$('figure');
  await figures[parseInt(index)].screenshot({path: out});
  console.log('wrote', out);
  await browser.close();
})();
