// Bundle the app (with preact and htm) into self-contained pages:
//
//   dist/psyfit-mobile.html   a complete document: open it in any browser, or
//                             host it anywhere as a single static file
//   dist/artifact.html        the same page as a Claude artifact body (no
//                             <html>/<head>/<body>; the publisher adds those)
//
// Nothing is fetched at run time, so the page keeps working offline once loaded.
//
//   node build.mjs            build both
//   node build.mjs --minify   smaller output

import { build } from 'esbuild';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
const TITLE = 'PsyFit Mobile';
const minify = process.argv.includes('--minify');

const result = await build({
  entryPoints: [join(root, 'src/main.js')],
  bundle: true,
  format: 'iife',
  target: ['es2020', 'safari15'],
  charset: 'utf8',
  legalComments: 'none',
  minify,
  write: false,
  logLevel: 'warning',
});

const js = result.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const css = (await readFile(join(root, 'src/app.css'), 'utf8')).replace(/<\/style/gi, '<\\/style');

const body = `<div id="app"></div>\n<script>\n${js}</script>\n`;

// The artifact publisher supplies the document skeleton, the viewport meta and
// the safe-area padding on :root.
const artifact = `<title>${TITLE}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
${css}</style>
${body}`;

const standalone = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="PsyFit">
<meta name="theme-color" content="#0b0e14">
<title>${TITLE}</title>
<style>
:root { box-sizing: border-box; padding: env(safe-area-inset-top, 0px) 0 env(safe-area-inset-bottom, 0px); }
${css}</style>
</head>
<body>
${body}</body>
</html>
`;

await mkdir(join(root, 'dist'), { recursive: true });
await writeFile(join(root, 'dist/psyfit-mobile.html'), standalone);
await writeFile(join(root, 'dist/artifact.html'), artifact);
const kb = (s) => `${(Buffer.byteLength(s) / 1024).toFixed(0)} kB`;
console.log(`dist/psyfit-mobile.html ${kb(standalone)}   dist/artifact.html ${kb(artifact)}${minify ? '   (minified)' : ''}`);
