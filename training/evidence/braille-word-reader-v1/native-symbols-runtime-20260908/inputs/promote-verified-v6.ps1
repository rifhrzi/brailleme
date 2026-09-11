$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$taskStageEvidence = (Resolve-Path -LiteralPath 'C:\bm-symbols-apk-20260908\evidence').Path
$expectedNewSha = '9afa6a1eb2983d0a92b4f1dcff3026f91d583bca02cda6d98cfaf91760a90259'
$expectedOldSha = '48b46c43b4b3aa776b07691480efd6e8e321dff80493a63183b27162e67a3e46'
$newSize = 133317765
$sourceApk = Join-Path $taskStageEvidence 'apk-integrity-v6\verified.apk'
$integrityPath = Join-Path $taskStageEvidence 'apk-integrity-v6\report.json'
$installPath = Join-Path $taskStageEvidence 'apk-integrity-v6\install.json'
$newApk = Join-Path $taskRoot 'output\android\BrailleMe-0.3.0-symbols-auto-frame-2026-09-09.apk'
$newShaFile = $newApk + '.sha256'
$oldApk = Join-Path $taskRoot 'output\android\BrailleMe-0.3.0-symbols-auto-frame-2026-09-08.apk'
$oldShaFile = $oldApk + '.sha256'
$deliveryPath = Join-Path $PSScriptRoot 'apk-delivery.json'
$archive = Join-Path $taskStageEvidence 'replaced-delivery-v2'

function Assert-Contained([string]$path, [string]$root) {
  $resolved = [IO.Path]::GetFullPath($path)
  $prefix = [IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
  if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Path outside intended directory: $resolved" }
}
function File-Sha([string]$path) {
  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Assert-File([string]$path) {
  $item = Get-Item -LiteralPath $path -Force
  if ($item.PSIsContainer -or (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) { throw "Expected ordinary file: $path" }
}
function Write-NewUtf8([string]$path, [string]$text) {
  if (Test-Path -LiteralPath $path) { throw "Refusing existing output: $path" }
  [IO.File]::WriteAllText($path, $text, [Text.UTF8Encoding]::new($false))
}

foreach ($path in @($newApk, $newShaFile, $oldApk, $oldShaFile, $deliveryPath)) { Assert-Contained $path $taskRoot }
foreach ($path in @($sourceApk, $integrityPath, $installPath, $archive)) { Assert-Contained $path $taskStageEvidence }
foreach ($path in @($sourceApk, $integrityPath, $installPath, $oldApk, $oldShaFile, $deliveryPath)) { Assert-File $path }
foreach ($path in @($newApk, $newShaFile, $archive)) { if (Test-Path -LiteralPath $path) { throw "New destination already exists: $path" } }
if ((File-Sha $sourceApk) -ne $expectedNewSha -or (Get-Item -LiteralPath $sourceApk).Length -ne $newSize) { throw 'Verified v6 source mismatch' }
if ((File-Sha $oldApk) -ne $expectedOldSha -or (Get-Item -LiteralPath $oldApk).Length -ne 133311473) { throw 'Old delivery differs from authorized v2 artifact; no files moved' }
$oldShaText = (Get-Content -LiteralPath $oldShaFile -Raw).Trim()
if (($oldShaText -split '\s+')[0] -ne $expectedOldSha) { throw 'Old SHA file mismatch' }
$previous = Get-Content -LiteralPath $deliveryPath -Raw | ConvertFrom-Json
if ($previous.sha256 -ne $expectedOldSha) { throw 'Old delivery metadata mismatch' }
$integrity = Get-Content -LiteralPath $integrityPath -Raw | ConvertFrom-Json
$install = Get-Content -LiteralPath $installPath -Raw | ConvertFrom-Json
if (-not $integrity.integrity_passed -or $integrity.apk.sha256 -ne $expectedNewSha -or $integrity.apk.bytes -ne $newSize) { throw 'Integrity report does not bind v6' }
if (-not $install.installedHashMatches -or $install.sha256 -ne $expectedNewSha -or $install.exitCode -ne 0) { throw 'Installation proof does not bind v6' }
$history = @($oldApk, $oldShaFile, $deliveryPath) | ForEach-Object {
  @{ originalPath = $_; archivePath = (Join-Path $archive ([IO.Path]::GetFileName($_))); bytes = (Get-Item -LiteralPath $_).Length; sha256 = (File-Sha $_) }
}

# Copy and verify the new deliverable before relocating the old delivery files.
Copy-Item -LiteralPath $sourceApk -Destination $newApk
if ((File-Sha $newApk) -ne $expectedNewSha -or (Get-Item -LiteralPath $newApk).Length -ne $newSize) { throw 'New APK copy failed hash/size check; old files left in place' }
Write-NewUtf8 $newShaFile ($expectedNewSha + '  ' + [IO.Path]::GetFileName($newApk) + "`n")
New-Item -ItemType Directory -Path $archive | Out-Null
foreach ($row in $history) {
  # Explicit ordinary files only, verified absolute containment; no recursive operation.
  Assert-Contained $row.originalPath $taskRoot
  Assert-Contained $row.archivePath $archive
  Assert-File $row.originalPath
  if ((File-Sha $row.originalPath) -ne $row.sha256) { throw 'Historical input changed before move' }
  if (Test-Path -LiteralPath $row.archivePath) { throw 'Archive collision' }
  Move-Item -LiteralPath $row.originalPath -Destination $row.archivePath
  if ((File-Sha $row.archivePath) -ne $row.sha256 -or (Get-Item -LiteralPath $row.archivePath).Length -ne $row.bytes) { throw 'Historical move hash/size mismatch' }
  if (Test-Path -LiteralPath $row.originalPath) { throw 'Old delivery location remains unexpectedly' }
}
$archiveIndex = @{ schemaVersion = 1; createdUtc = [DateTime]::UtcNow.ToString('o'); status = 'historical_v2_delivery_preserved'; files = $history; removedOldDeliveryLocations = @($oldApk, $oldShaFile); note = 'Native Move-Item of three explicitly verified files; no recursive delete or move.' }
$archiveIndexPath = Join-Path $archive 'archive-index.json'
Write-NewUtf8 $archiveIndexPath (($archiveIndex | ConvertTo-Json -Depth 8) + "`n")
$delivery = @{
  schemaVersion = 2; createdUtc = [DateTime]::UtcNow.ToString('o'); sourceApk = $sourceApk; apk = $newApk;
  sha256 = $expectedNewSha; bytes = $newSize; sha256File = $newShaFile;
  status = 'experimental_runtime_measurement_in_progress'; physicalCameraValidated = $false; runtimeMeasurementsComplete = $false;
  integrityReport = $integrityPath; integrityReportSha256 = (File-Sha $integrityPath);
  installedApkProof = $installPath; installedApkProofSha256 = (File-Sha $installPath); installedHashMatches = $true;
  previousDeliveryArchive = $archiveIndexPath; previousDeliveryArchiveSha256 = (File-Sha $archiveIndexPath);
  limitations = @('Internal Android Debug signature; not a production release.', 'Capital recognition and whole-word localization remain experimental.', 'The 600-second emulator measurement is in progress; no physical-camera validation.');
  promotionScriptSha256 = (File-Sha $PSCommandPath)
}
Write-NewUtf8 $deliveryPath (($delivery | ConvertTo-Json -Depth 8) + "`n")
if ((File-Sha $newApk) -ne $expectedNewSha -or (File-Sha $sourceApk) -ne $expectedNewSha) { throw 'Final source/destination hash changed' }
Write-Output ($delivery | ConvertTo-Json -Depth 8)
