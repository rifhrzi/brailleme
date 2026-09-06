$ErrorActionPreference = 'Stop'
$testOriginalLocation = Get-Location
$testDirectory = Join-Path $PSScriptRoot 'release-helper-integrity/powershell-mocked-run'
if (Test-Path -LiteralPath $testDirectory) { throw 'Use a new mocked test directory' }
New-Item -ItemType Directory -Path $testDirectory | Out-Null
$global:testChecks = @()
$global:testPythonCalls = @()
function global:npm.cmd {
  param([string]$Verb, [string]$Check)
  if ($Verb -ne 'run') { throw 'Unexpected npm command' }
  $global:testChecks += $Check
  $global:LASTEXITCODE = 0
  Write-Output "MOCK ONLY: $Check"
}
function global:Test-Python {
  $global:testPythonCalls += ,@($args)
  $testSnapshotIndex = [Array]::IndexOf($args, '--snapshot')
  $testIdentityIndex = [Array]::IndexOf($args, '--verification-id')
  if ($testSnapshotIndex -lt 0 -or $testIdentityIndex -lt 0) { throw 'Missing snapshot arguments' }
  $testSnapshotPath = $args[$testSnapshotIndex + 1]
  $testIdentity = $args[$testIdentityIndex + 1]
  if ($global:testPythonCalls.Count -eq 2) {
    if ($global:testChecks.Count -ne 6 -or [Array]::IndexOf($args, '--compare') -lt 0 -or [Array]::IndexOf($args, '--checks') -lt 0) { throw 'Final snapshot is not tied to all checks' }
  }
  @{ scope = 'MOCK ONLY; no real source/model/config/check was executed'; verificationId = $testIdentity } | ConvertTo-Json | Set-Content -LiteralPath $testSnapshotPath -Encoding UTF8
  $global:LASTEXITCODE = 0
}
try {
  Set-Location -LiteralPath $testDirectory
  . (Join-Path $PSScriptRoot 'verify-source.ps1') -PythonExecutable 'Test-Python'
  $testEvidence = Join-Path $testDirectory 'output/word-scan-20260905'
  $testResults = Get-Content -LiteralPath (Join-Path $testEvidence 'source-checks.json') -Raw | ConvertFrom-Json
  $testBefore = Join-Path $testEvidence 'source-snapshot-before.json'
  $testBeforeHash = (Get-FileHash -LiteralPath $testBefore -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($testResults.Count -ne 6 -or $global:testPythonCalls.Count -ne 2) { throw 'Unexpected phase count' }
  foreach ($testRow in $testResults) {
    if ($testRow.exitCode -ne 0 -or $testRow.verificationId -ne $taskVerificationId -or $testRow.sourceSnapshotSha256 -ne $testBeforeHash) { throw 'Unbound check row' }
  }
  Write-Output 'PASS: actual PowerShell control flow invoked two snapshots and six mocked checks; every row is bound to the before snapshot and run id.'
} finally {
  Set-Location -LiteralPath $testOriginalLocation
}
