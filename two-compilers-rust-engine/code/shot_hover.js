// shot_hover.js URL OUT.png INDEX N [SCHEME]: hovers the N-th mark of the INDEX-th figure, prints the tooltip's text and
// any error of the page, and takes a screenshot of the window with the tooltip in it.
const puppeteer = require('/home/dev/.npm/_npx/d62b6517736c1e35/node_modules/puppeteer');
(async () => {
  const [url, out, index, n, scheme] = process.argv.slice(2);
  const browser = await puppeteer.launch({executablePath: '/home/dev/.cache/puppeteer/chrome-headless-shell/linux-148.0.7778.97/chrome-headless-shell-linux64/chrome-headless-shell', args: ['--no-sandbox']});
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('PAGE ERROR', e.message));
  page.on('console', (m) => { if (m.type() === 'error') console.log('CONSOLE ERROR', m.text()); });
  await page.setViewport({width: 1000, height: 1100, deviceScaleFactor: 1});
  if (scheme && scheme !== 'light') await page.emulateMediaFeatures([{name: 'prefers-color-scheme', value: scheme}]);
  await page.goto(url);
  const figures = await page.$$('figure');
  const figure = figures[parseInt(index)];
  await figure.evaluate((el) => el.scrollIntoView({block: 'center'}));
  const marks = await figure.$$('.hit, svg[data-xh]');
  const box = await marks[parseInt(n)].boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await new Promise((r) => setTimeout(r, 200));
  const tip = await page.evaluate(() => { const t = document.getElementById('tip'); return t && t.style.display !== 'none' ? t.innerText : null; });
  console.log('marks in the figure:', marks.length, 'tooltip:', JSON.stringify(tip));
  await page.screenshot({path: out});
  console.log('wrote', out);
  await browser.close();
})();
