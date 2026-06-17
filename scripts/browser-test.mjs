// Headless-browser proof for the eyebrowse WebGPU runtime.
// Serves web/ over http, drives the system Chrome with WebGPU enabled, runs run_add_demo(),
// and asserts the GPU-computed sum equals 231. Exits non-zero on any failure.
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const here = fileURLToPath(new URL('.', import.meta.url));
const WEB_DIR = join(here, '..', 'web');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
};

const server = http.createServer(async (req, res) => {
  try {
    let p = normalize(decodeURIComponent(req.url.split('?')[0]));
    if (p === '/' || p === '\\') p = '/index.html';
    const data = await readFile(join(WEB_DIR, p));
    res.writeHead(200, { 'content-type': MIME[extname(p)] || 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('not found');
  }
});
await new Promise((r) => server.listen(0, r));
const port = server.address().port;
const url = `http://localhost:${port}/index.html`;

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: [
    '--enable-unsafe-webgpu',
    '--use-angle=metal',
    '--no-sandbox',
    '--enable-features=WebGPU',
  ],
});
const page = await browser.newPage();
page.on('console', (m) => console.log('[page]', m.text()));
page.on('pageerror', (e) => console.log('[pageerror]', e.message));

let exitCode = 1;
try {
  await page.goto(url, { waitUntil: 'load', timeout: 60000 });
  await page.waitForFunction(
    'window.__RESULT__ !== undefined || window.__ERROR__ !== undefined',
    { timeout: 60000 },
  );
  const result = await page.evaluate('window.__RESULT__');
  const err = await page.evaluate('window.__ERROR__');
  if (err) {
    console.error('browser error:', err);
  } else if (typeof result === 'number' && Math.abs(result - 231) < 1e-3) {
    console.log('browser add OK:', result);
    exitCode = 0;
  } else {
    console.error('browser add WRONG:', result);
  }
} catch (e) {
  console.error('test failed:', e.message);
} finally {
  await browser.close();
  server.close();
  process.exit(exitCode);
}
