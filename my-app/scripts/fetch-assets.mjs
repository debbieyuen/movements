// Fetches the MediaPipe runtime assets into public/ so the app has ZERO
// runtime CDN dependency (the old @mediapipe/pose loaded WASM from jsDelivr
// at capture time -- a network hiccup killed pose tracking).
//
//   public/mediapipe/wasm/   <- copied from node_modules/@mediapipe/tasks-vision
//   public/models/pose_landmarker_full.task  <- downloaded from Google storage
//
// Runs automatically via the package.json `postinstall` hook; both output
// directories are gitignored. Safe to re-run; skips files that already exist.

import { createWriteStream, existsSync, mkdirSync, copyFileSync, readdirSync } from 'node:fs';
import { get } from 'node:https';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const wasmSrc = join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const wasmDst = join(root, 'public', 'mediapipe', 'wasm');
const modelDst = join(root, 'public', 'models', 'pose_landmarker_full.task');

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';

mkdirSync(wasmDst, { recursive: true });
for (const f of readdirSync(wasmSrc)) {
  const dst = join(wasmDst, f);
  if (!existsSync(dst)) {
    copyFileSync(join(wasmSrc, f), dst);
    console.log(`copied ${f}`);
  }
}

function download(url, dst, redirects = 0) {
  return new Promise((resolve, reject) => {
    get(url, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        if (redirects > 5) return reject(new Error('too many redirects'));
        return resolve(download(res.headers.location, dst, redirects + 1));
      }
      if (res.statusCode !== 200) {
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
      }
      mkdirSync(dirname(dst), { recursive: true });
      const out = createWriteStream(dst);
      res.pipe(out);
      out.on('finish', () => out.close(resolve));
      out.on('error', reject);
    }).on('error', reject);
  });
}

if (!existsSync(modelDst)) {
  console.log('downloading pose_landmarker_full.task (~9 MB)...');
  try {
    await download(MODEL_URL, modelDst);
    console.log('model downloaded');
  } catch (err) {
    console.warn(
      `WARNING: could not download the pose model (${err.message}). ` +
      'Pose tracking will not work until you re-run: node scripts/fetch-assets.mjs');
  }
} else {
  console.log('pose model already present');
}
