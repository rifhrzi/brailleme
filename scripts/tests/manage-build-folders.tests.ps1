$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$manager = Join-Path $repo 'scripts/manage-build-folders.ps1'
$testRoot = Join-Path $repo ('output/build-folder-tests/' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null
$passed = New-Object 'System.Collections.Generic.List[string]'
function Assert($Condition, [string]$Message) { if (!$Condition) { throw $Message } }
function Case-Root([string]$Name) {
  $dir = Join-Path $testRoot $Name
  New-Item -ItemType Directory -Path $dir | Out-Null
  return $dir
}
function Run([string]$Root, [string]$Action, [string]$Name) {
  & $manager -BuildRoot $Root -ArchiveRoot (Join-Path $Root 'archives') -Action $Action -Path (Join-Path $Root $Name)
}
function Refuses([scriptblock]$Command, [string]$Pattern) {
  $failure = $null
  try { & $Command | Out-Null } catch { $failure = $_.Exception.Message }
  Assert ($failure -and $failure -match $Pattern) "Expected rejection '$Pattern'; got '$failure'"
}

$root = Case-Root 'rotation'
$first = Run $root Create 'bm-build-first'
$second = Run $root Create 'bm-build-second'
# Deterministic age order, including on coarse-resolution file systems.
(Get-Item -LiteralPath $first).CreationTimeUtc = [DateTime]::UtcNow.AddDays(-2)
(Get-Item -LiteralPath $second).CreationTimeUtc = [DateTime]::UtcNow.AddDays(-1)
foreach ($relative in @('evidence/result.json','android/app/build/outputs/apk/release/app.apk','node_modules/cache.bin','android/app/build/intermediates/cache.bin')) {
  $file = Join-Path $first $relative
  New-Item -ItemType Directory -Path (Split-Path $file) -Force | Out-Null
  Set-Content -LiteralPath $file -Value $relative -Encoding UTF8
}
$protected = Join-Path $root 'bm-datasets'
New-Item -ItemType Directory -Path $protected | Out-Null
$third = Run $root Create 'bm-build-third'
Assert (!(Test-Path -LiteralPath $first)) 'Oldest stage was not removed'
Assert ((Test-Path -LiteralPath $second) -and (Test-Path -LiteralPath $third) -and (Test-Path -LiteralPath $protected)) 'Survivors/protected directory missing'
$status = Run $root Status 'unused' | ConvertFrom-Json
Assert ($status.count -eq 2 -and $status.withinLimit) 'Two-folder cap failed'
$manifestFile = Get-ChildItem -LiteralPath (Join-Path $root 'archives/bm-build-first') -Filter manifest.json -Recurse
$manifest = Get-Content -LiteralPath $manifestFile.FullName -Raw | ConvertFrom-Json
Assert ($manifest.files.relativePath -contains 'evidence/result.json') 'Evidence omitted'
Assert ($manifest.files.relativePath -contains 'android/app/build/outputs/apk/release/app.apk') 'APK omitted'
Assert (!($manifest.files.relativePath -match 'cache.bin')) 'Regenerable cache archived'
foreach ($file in $manifest.files) {
  Assert ((Get-FileHash -LiteralPath (Join-Path $manifest.archiveRoot $file.blob)).Hash -eq $file.sha256) 'Archived hash mismatch'
}
$passed.Add('third creation archives verified APK/evidence, excludes caches, deletes oldest first, keeps two and protects datasets')
Refuses { Run $root Create 'bm-build-second' } 'Destination already exists'
$passed.Add('existing destination cannot be overwritten')
Refuses { Run $root Create '../outside' } 'Not a permitted direct build child'
$passed.Add('path escape is rejected')

$root = Case-Root 'unregistered'
Run $root Create 'bm-build-one' | Out-Null
New-Item -ItemType Directory -Path (Join-Path $root 'bm-build-unknown') | Out-Null
Refuses { Run $root Create 'bm-build-three' } 'Unregistered build'
Assert (!(Test-Path -LiteralPath (Join-Path $root 'bm-build-three'))) 'Unknown stage allowed third slot'
$passed.Add('unregistered stage blocks rotation')
Refuses { Run $root Adopt 'bm-build-unknown' } 'package.json'
$passed.Add('adoption requires project identity')

$root = Case-Root 'corrupt-archive'
$first = Run $root Create 'bm-build-one'
Run $root Create 'bm-build-two' | Out-Null
(Get-Item -LiteralPath $first).CreationTimeUtc = [DateTime]::UtcNow.AddDays(-2)
$manifestPath = Run $root Archive 'bm-build-one'
$record = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
Set-Content -LiteralPath (Join-Path $record.archiveRoot $record.files[0].blob) -Value 'corrupt'
Refuses { Run $root Create 'bm-build-three' } 'Archive/source changed or corrupt'
Assert ((Test-Path -LiteralPath $first) -and !(Test-Path -LiteralPath (Join-Path $root 'bm-build-three'))) 'Corrupt archive allowed deletion/creation'
$passed.Add('corrupt archive prevents deletion and third creation')

$root = Case-Root 'junction'
$stage = Run $root Create 'bm-build-one'
$outside = Join-Path $root 'protected'
New-Item -ItemType Directory -Path $outside | Out-Null
New-Item -ItemType Junction -Path (Join-Path $stage 'linked') -Target $outside | Out-Null
Refuses { Run $root Archive 'bm-build-one' } 'Reparse point inside build'
Assert (Test-Path -LiteralPath $outside) 'Junction target affected'
$passed.Add('junction is rejected without touching its target')

$root = Case-Root 'manager-lock'
$lock = [IO.File]::Open((Join-Path $repo 'artifacts/build-folder-manager.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
try { Refuses { Run $root Create 'bm-build-one' } 'being used by another process' } finally { $lock.Dispose() }
Assert (!(Test-Path -LiteralPath (Join-Path $root 'bm-build-one'))) 'Concurrent manager created a stage'
$passed.Add('manager lock blocks concurrent creation even with a different archive root')

$root = Case-Root 'build-locks'
$first = Run $root Create 'bm-build-one'
$second = Run $root Create 'bm-build-two'
$handles = @()
try {
  foreach ($stage in @($first,$second)) {
    $lockDir = Join-Path $stage '.gradle'
    New-Item -ItemType Directory -Path $lockDir | Out-Null
    $handles += [IO.File]::Open((Join-Path $lockDir 'build.lock'),[IO.FileMode]::Create,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
  }
  Refuses { Run $root Create 'bm-build-three' } 'No safely removable build folder'
  Assert ((Test-Path -LiteralPath $first) -and (Test-Path -LiteralPath $second) -and !(Test-Path -LiteralPath (Join-Path $root 'bm-build-three'))) 'Active build was removed'
} finally { foreach ($handle in $handles) { $handle.Dispose() } }
$passed.Add('active build locks prevent rotation')

$root = Case-Root 'shared-log'
$stage = Run $root Create 'bm-build-one'
$logPath = Join-Path $stage 'emulator.log'
$writer = [IO.File]::Open($logPath,[IO.FileMode]::Create,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try {
  $bytes = [Text.Encoding]::UTF8.GetBytes('finished emulator log')
  $writer.Write($bytes,0,$bytes.Length)
  $writer.Flush()
  $snapshot = Run $root Archive 'bm-build-one'
  $record = Get-Content -LiteralPath $snapshot -Raw | ConvertFrom-Json
  $logRecord = $record.files | Where-Object relativePath -EQ 'emulator.log'
  Assert ($logRecord.bytes -eq $bytes.Length) 'Shared log omitted or truncated'
  Assert ([IO.File]::ReadAllText((Join-Path $record.archiveRoot $logRecord.blob)) -eq 'finished emulator log') 'Shared log archive differs'
} finally { $writer.Dispose() }
$passed.Add('stable redirected log can be archived with a retained shared writer handle')

$result = @{ passed=$passed.Count; cases=@($passed.ToArray()); fixtureRoot=$testRoot; productionBuildFoldersTouched=$false }
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $testRoot 'result.json') -Encoding UTF8
$result | ConvertTo-Json -Depth 5
