param([string]$PythonExecutable = 'C:\bm-envs\angelina-benchmark\Scripts\python.exe')
$ErrorActionPreference = 'Stop'
$env:APP_ENV = 'preview'
$env:CI = '1'
$taskEvidenceDirectory = Join-Path (Get-Location) 'output/word-scan-20260905'
New-Item -ItemType Directory -Path $taskEvidenceDirectory -Force | Out-Null
$taskVerificationId = [Guid]::NewGuid().ToString('D')
$taskSnapshotHelper = Join-Path $PSScriptRoot 'stage_source.py'
$taskBeforeSnapshot = Join-Path $taskEvidenceDirectory 'source-snapshot-before.json'
$taskAfterSnapshot = Join-Path $taskEvidenceDirectory 'source-snapshot-after.json'
$taskChecksPath = Join-Path $taskEvidenceDirectory 'source-checks.json'
& $PythonExecutable $taskSnapshotHelper --snapshot $taskBeforeSnapshot --verification-id $taskVerificationId
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$taskBeforeHash = (Get-FileHash -LiteralPath $taskBeforeSnapshot -Algorithm SHA256).Hash.ToLowerInvariant()
$taskChecks = @('model:verify', 'release:config:verify', 'lint', 'typecheck', 'test:coverage', 'export:android')
$taskResults = @()
foreach ($taskCheck in $taskChecks) {
  $taskStart = Get-Date
  $taskLog = Join-Path $taskEvidenceDirectory ($taskCheck.Replace(':', '-') + '.log')
  $ErrorActionPreference = 'Continue'
  & npm.cmd run $taskCheck 2>&1 | Tee-Object -FilePath $taskLog
  $taskExit = $LASTEXITCODE
  $ErrorActionPreference = 'Stop'
  $taskResults += [pscustomobject]@{ check = $taskCheck; exitCode = $taskExit; verificationId = $taskVerificationId; sourceSnapshotSha256 = $taskBeforeHash; started = $taskStart.ToUniversalTime().ToString('o'); finished = (Get-Date).ToUniversalTime().ToString('o') }
  $taskResults | ConvertTo-Json | Set-Content -LiteralPath $taskChecksPath -Encoding UTF8
  if ($taskExit -ne 0) { exit $taskExit }
}
& $PythonExecutable $taskSnapshotHelper --snapshot $taskAfterSnapshot --verification-id $taskVerificationId --checks $taskChecksPath --compare $taskBeforeSnapshot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
