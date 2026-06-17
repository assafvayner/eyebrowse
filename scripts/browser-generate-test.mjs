// Headless-browser proof for the eyebrowse WebGPU LLM runtime.
// Serves web/ at / and the Qwen3 model dir at /model/, drives system Chrome with WebGPU enabled,
// runs Qwen3-0.6B greedy generation in the tab, and compares the ids to the HF golden.
// Exits 0 on pass (first id == 12095 AND >= 18/20 ids match), non-zero otherwise.
import http from 'node:http';
import fs from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const here = fileURLToPath(new URL('.', import.meta.url));
const REPO = join(here, '..');
const WEB_DIR = join(REPO, 'web');
const MODEL_DIR = join(REPO, 'models', 'qwen3-0.6b');
const GOLDEN_PATH = join(REPO, 'golden', 'qwen3-golden.json');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.safetensors': 'application/octet-stream',
};

const server = http.createServer(async (req, res) => {
  try {
    let p = normalize(decodeURIComponent(req.url.split('?')[0]));
    // Model files: stream from the model dir so we never buffer the 1.4GB blob.
    if (p.startsWith('/model/')) {
      const rel = p.slice('/model/'.length);
      const filePath = join(MODEL_DIR, rel);
      if (!filePath.startsWith(MODEL_DIR)) {
        res.writeHead(403);
        res.end('forbidden');
        return;
      }
      const st = await stat(filePath);
      res.writeHead(200, {
        'content-type': MIME[extname(filePath)] || 'application/octet-stream',
        'content-length': st.size,
      });
      fs.createReadStream(filePath).pipe(res);
      return;
    }
    // Everything else: small static files under web/.
    if (p === '/' || p === '\\') p = '/qwen3.html';
    const filePath = join(WEB_DIR, p);
    if (!filePath.startsWith(WEB_DIR)) {
      res.writeHead(403);
      res.end('forbidden');
      return;
    }
    const data = await readFile(filePath);
    res.writeHead(200, { 'content-type': MIME[extname(p)] || 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('not found');
  }
});

await new Promise((r) => server.listen(0, r));
const port = server.address().port;
const url = `http://localhost:${port}/qwen3.html`;

const golden = JSON.parse(await readFile(GOLDEN_PATH, 'utf8'));
const goldenIds = golden.greedy_continuation_ids;

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-sandbox'],
});
const page = await browser.newPage();
page.on('console', (m) => console.log('[page]', m.text()));
page.on('pageerror', (e) => console.log('[pageerror]', e.message));

let exitCode = 1;
const t0 = Date.now();
try {
  await page.goto(url, { waitUntil: 'load', timeout: 60000 });
  await page.waitForFunction(
    'window.__RESULT__ !== undefined || window.__ERROR__ !== undefined',
    { timeout: 300000 },
  );
  const result = await page.evaluate('window.__RESULT__');
  const err = await page.evaluate('window.__ERROR__');

  if (err) {
    console.error('\nbrowser error:\n' + err);
  } else if (Array.isArray(result)) {
    const n = Math.min(result.length, goldenIds.length);
    let leading = 0;
    for (let i = 0; i < n && result[i] === goldenIds[i]; i++) leading++;
    let matches = 0;
    for (let i = 0; i < n; i++) if (result[i] === goldenIds[i]) matches++;

    console.log('\n=== Qwen3-0.6B browser generation vs HF golden ===');
    console.log('got:    [' + result.join(', ') + ']');
    console.log('golden: [' + goldenIds.join(', ') + ']');
    console.log(`leading-match: ${leading}/${goldenIds.length}`);
    console.log(`total matches: ${matches}/${goldenIds.length}`);
    console.log(`first id: got ${result[0]}, expected ${goldenIds[0]}`);
    console.log(`wall time (nav -> result): ${((Date.now() - t0) / 1000).toFixed(2)}s`);

    if (result[0] === goldenIds[0] && matches >= 18) {
      console.log('PASS');
      exitCode = 0;
    } else {
      console.log('FAIL');
    }
  } else {
    console.error('no result produced:', result);
  }
} catch (e) {
  console.error('test failed:', e.message);
} finally {
  await browser.close();
  server.close();
  process.exit(exitCode);
}
