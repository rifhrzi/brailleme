const fs = require('node:fs');
const path = require('node:path');
const cp = require('node:child_process');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '../..');
const stage = 'C:/bm-symbols-apk-20260908';
const files = [...new Set(cp.execFileSync('git', ['ls-files', '-c', '-o', '--exclude-standard', '-z'], { cwd: root, maxBuffer: 16 * 1024 * 1024 }).toString().split('\0').filter(Boolean))];
const hashes = {};
for (const file of files) {
  if (/^(training|docs|output|artifacts|\.github)\//.test(file)) continue;
  const src = path.join(root, file), dest = path.join(stage, file);
  if (!fs.existsSync(src) || !fs.statSync(src).isFile()) continue;
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const bytes = fs.readFileSync(src);
  if (!fs.existsSync(dest) || !bytes.equals(fs.readFileSync(dest))) fs.writeFileSync(dest, bytes);
  hashes[file] = crypto.createHash('sha256').update(bytes).digest('hex');
}
fs.writeFileSync(path.join(__dirname, 'staged-source.json'), JSON.stringify({ stage, files: hashes }, null, 2));
console.log(`Staged ${Object.keys(hashes).length} source files, including existing camera changes.`);
