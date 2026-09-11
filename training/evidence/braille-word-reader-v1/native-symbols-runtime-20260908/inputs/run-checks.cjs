const fs = require('node:fs');
const path = require('node:path');
const cp = require('node:child_process');
const root = path.resolve(__dirname, '../..');
const directory = path.join(__dirname, `checks-${Date.now()}`);
fs.mkdirSync(directory);
const checks = [];
for (const name of ['model:verify', 'release:config:verify', 'lint', 'typecheck', 'test:coverage']) {
  const started = new Date().toISOString();
  const result = cp.spawnSync('npm.cmd', ['run', name], {
    cwd: root, shell: true, windowsHide: true, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024,
  });
  const file = name.replaceAll(':', '-') + '.log';
  fs.writeFileSync(path.join(directory, file), (result.stdout || '') + (result.stderr || ''));
  checks.push({ name, started, completed: new Date().toISOString(), exitCode: result.status, log: file });
  fs.writeFileSync(path.join(directory, 'checks.json'), JSON.stringify(checks, null, 2));
  console.log(`${name}: ${result.status === 0 ? 'PASS' : 'FAIL'} (${file})`);
  if (result.status !== 0) { console.log(result.stderr || result.stdout); process.exit(result.status || 1); }
}
console.log(`Evidence: ${directory}`);
