param(
  [ValidateSet('Status','Adopt','Ensure','Create','Archive')][string]$Action = 'Status',
  [string]$Path,
  [string]$BuildRoot = 'C:\',
  [string]$ArchiveRoot
)
$ErrorActionPreference = 'Stop'
$wordRepo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$wordBuildRoot = [IO.Path]::GetFullPath($BuildRoot).TrimEnd('\') + '\'
if (!$ArchiveRoot) { $ArchiveRoot = Join-Path $wordRepo 'artifacts\build-archives' }
$wordArchiveRoot = [IO.Path]::GetFullPath($ArchiveRoot).TrimEnd('\')
$wordMarker = '.brailleme-build.json'
$wordLimit = 2

function Assert-NoLink([string]$Target) {
  $cursor = $Target
  while ($cursor) {
    if (Test-Path -LiteralPath $cursor) {
      $entry = Get-Item -LiteralPath $cursor -Force
      if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point rejected: $cursor" }
    }
    $parent = Split-Path -Parent $cursor
    if ($parent -eq $cursor) { break }
    $cursor = $parent
  }
}
function Resolve-Stage([string]$Target) {
  if (!$Target -or ![IO.Path]::IsPathRooted($Target)) { throw 'An absolute build path is required.' }
  $full = [IO.Path]::GetFullPath($Target).TrimEnd('\')
  $parent = [IO.Path]::GetFullPath((Split-Path -Parent $full)).TrimEnd('\') + '\'
  $name = Split-Path -Leaf $full
  if ($parent -ne $wordBuildRoot -or $name -notmatch '^bm-(?:apk|symbols-apk|build)-[A-Za-z0-9][A-Za-z0-9-]*$') { throw "Not a permitted direct build child: $full" }
  if ($full -eq $wordRepo -or $wordRepo.StartsWith($full + '\', [StringComparison]::OrdinalIgnoreCase) -or $wordArchiveRoot -eq $full -or $wordArchiveRoot.StartsWith($full + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Repository/archive cannot be a deletion target.' }
  Assert-NoLink $full
  return $full
}
function Read-Marker([string]$Stage) {
  $file = Join-Path $Stage $wordMarker
  Assert-NoLink $file
  if (!(Test-Path -LiteralPath $file)) { throw "Unregistered build; explicitly Adopt after inspection: $Stage" }
  $record = Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json
  if ($record.project -ne 'brailleme' -or $record.repository -ne $wordRepo -or $record.path -ne $Stage) { throw "Ownership mismatch: $Stage" }
  [void][DateTime]::Parse($record.createdUtc)
  return $record
}
function Get-Stages {
  @(Get-ChildItem -LiteralPath $wordBuildRoot -Directory -Force | Where-Object Name -Match '^bm-(?:apk|symbols-apk|build)-' | ForEach-Object {
    $stage = Resolve-Stage $_.FullName
    [pscustomobject]@{ path=$stage; registered=(Test-Path -LiteralPath (Join-Path $stage $wordMarker)); createdUtc=$_.CreationTimeUtc.ToString('o') }
  })
}
function Get-Files([string]$Stage, [bool]$PreservedOnly) {
  $stack = New-Object 'System.Collections.Generic.Stack[string]'
  $stack.Push($Stage)
  while ($stack.Count) {
    $folder = $stack.Pop()
    foreach ($item in Get-ChildItem -LiteralPath $folder -Force) {
      if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point inside build: $($item.FullName)" }
      $relative = $item.FullName.Substring($Stage.Length + 1).Replace('\','/')
      if ($item.PSIsContainer) {
        if ($PreservedOnly) {
          if ($item.Name -in @('node_modules','.gradle','.cxx','.kotlin','.git')) { continue }
          if ($relative -match '(^|/)android/(?:[^/]+/)*build$') {
            # Preserve distributable APK/AAB, mapping and build-output metadata.
            if ($relative -eq 'android/app/build') {
              $outputs = Join-Path $item.FullName 'outputs'
              if (Test-Path -LiteralPath $outputs) { Assert-NoLink $outputs; $stack.Push($outputs) }
            }
            continue
          }
        }
        $stack.Push($item.FullName)
      } else { $item }
    }
  }
}
function Save-Json([string]$File, $Value) {
  $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $File -Encoding UTF8
}
function Hash-Source([string]$File) {
  # Redirected emulator logs may retain a shared writer handle after shutdown.
  # Allow reading; before/after hashes and metadata still reject changing files.
  $stream = [IO.File]::Open($File,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
  $sha = [Security.Cryptography.SHA256]::Create()
  try { return [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
  finally { $sha.Dispose(); $stream.Dispose() }
}
function Copy-Source([string]$Source, [string]$Destination) {
  $inputStream = [IO.File]::Open($Source,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
  try {
    $outputStream = [IO.File]::Open($Destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try { $inputStream.CopyTo($outputStream) } finally { $outputStream.Dispose() }
  } finally { $inputStream.Dispose() }
}
function Archive-Stage([string]$Stage) {
  [void](Read-Marker $Stage)
  $files = @(Get-Files $Stage $true | Sort-Object FullName)
  $snapshot = Join-Path $wordArchiveRoot ((Split-Path -Leaf $Stage) + '\' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8))
  Assert-NoLink $snapshot
  New-Item -ItemType Directory -Path $snapshot | Out-Null
  $blobs = Join-Path $wordArchiveRoot 'blobs'
  New-Item -ItemType Directory -Path $blobs -Force | Out-Null
  Assert-NoLink $blobs
  $rows = @()
  foreach ($file in $files) {
    $before = Get-Item -LiteralPath $file.FullName -Force
    $hash = Hash-Source $file.FullName
    $blob = Join-Path $blobs $hash
    if (!(Test-Path -LiteralPath $blob)) { Copy-Source $file.FullName $blob }
    Assert-NoLink $blob
    $after = Get-Item -LiteralPath $file.FullName -Force
    if ($before.Length -ne $after.Length -or $before.LastWriteTimeUtc -ne $after.LastWriteTimeUtc -or (Hash-Source $file.FullName) -ne $hash -or (Get-FileHash -LiteralPath $blob -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hash) { throw "Archive/source changed or corrupt: $($file.FullName)" }
    $rows += [pscustomobject]@{ relativePath=$file.FullName.Substring($Stage.Length+1).Replace('\','/'); bytes=$after.Length; sha256=$hash; blob=('blobs/' + $hash) }
  }
  $again = @(Get-Files $Stage $true | Sort-Object FullName)
  if (($again.FullName -join "`n") -ne ($files.FullName -join "`n")) { throw 'Source inventory changed while archiving.' }
  $manifest = Join-Path $snapshot 'manifest.json'
  Save-Json $manifest @{ schemaVersion=1; project='brailleme'; source=$Stage; createdUtc=[DateTime]::UtcNow.ToString('o'); archiveRoot=$wordArchiveRoot; verified=$true; files=$rows; fileCount=$rows.Count; logicalBytes=($rows | Measure-Object bytes -Sum).Sum; excludedRegenerable=@('node_modules','.gradle','.cxx','.kotlin','android/**/build except app/build/outputs','.git'); restore='For each files entry copy archiveRoot/blob to a new destination/relativePath; verify SHA-256 first. Keep blobs and all manifests together.' }
  Write-Host "Archived and verified $($rows.Count) files: $manifest"
  return $manifest
}
function Assert-Idle([string]$Stage) {
  $normalized = $Stage.Replace('\','/')
  $users = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.CommandLine -and $_.Name -match '^(java|javaw|node|ninja|cmake|qemu.*|emulator|gradle).*' -and $_.CommandLine.Replace('\','/').IndexOf($normalized,[StringComparison]::OrdinalIgnoreCase) -ge 0
  })
  if ($users.Count) { throw "Build is in use; stop its owned processes first: $Stage (PID $($users.ProcessId -join ', '))" }
  $worktrees = & git -C $wordRepo worktree list --porcelain
  if ($LASTEXITCODE -ne 0) { throw 'Cannot verify worktree registration.' }
  if (@($worktrees | Where-Object { $_ -like 'worktree *' -and $_.Substring(9).Replace('\','/').TrimEnd('/') -eq $normalized }).Count) { throw 'Registered worktree: use an explicit worktree removal workflow.' }
  # Java/Gradle may reference its project only in its working directory.
  $locks = @(Get-Files $Stage $false | Where-Object { $_.Extension -eq '.lock' -and $_.FullName -match '\\(?:\.gradle|\.cxx)\\' })
  foreach ($file in $locks) {
    $handle = [IO.File]::Open($file.FullName,[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    $handle.Dispose()
  }
}
function Remove-ArchivedStage([string]$Stage) {
  $safe = Resolve-Stage $Stage
  [void](Read-Marker $safe)
  Assert-Idle $safe
  $manifest = Archive-Stage $safe
  # Recheck the full tree, ownership, active processes and protected artifacts immediately before deletion.
  $safe = Resolve-Stage $safe
  [void](Read-Marker $safe)
  Assert-Idle $safe
  $record = Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json
  $currentPaths = @(Get-Files $safe $true | ForEach-Object { $_.FullName.Substring($safe.Length+1).Replace('\','/') } | Sort-Object)
  if (($currentPaths -join "`n") -ne (($record.files.relativePath | Sort-Object) -join "`n")) { throw 'Pre-delete source inventory changed.' }
  foreach ($file in $record.files) {
    if ((Get-FileHash -LiteralPath (Join-Path $safe $file.relativePath) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256 -or (Get-FileHash -LiteralPath (Join-Path $wordArchiveRoot $file.blob) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) { throw 'Pre-delete archive verification failed.' }
  }
  Remove-Item -LiteralPath $safe -Recurse -Force
  if (Test-Path -LiteralPath $safe) { throw 'Old build still exists; no new build may be created.' }
  Save-Json (Join-Path (Split-Path $manifest) 'rotation.json') @{ deletedPath=$safe; deletedUtc=[DateTime]::UtcNow.ToString('o'); archiveManifest=$manifest; archiveSha256=(Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant(); removalVerified=$true }
}

Assert-NoLink $wordBuildRoot
Assert-NoLink $wordArchiveRoot
New-Item -ItemType Directory -Path $wordArchiveRoot -Force | Out-Null
# Serialize all managed creation/rotation; failure to acquire never creates a third slot.
$wordStateRoot = Join-Path $wordRepo 'artifacts'
Assert-NoLink $wordStateRoot
New-Item -ItemType Directory -Path $wordStateRoot -Force | Out-Null
$wordLockPath = Join-Path $wordStateRoot 'build-folder-manager.lock'
Assert-NoLink $wordLockPath
$wordLock = [IO.File]::Open($wordLockPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
try {
  if ($Action -eq 'Status') {
    $stages = @(Get-Stages)
    @{ maxFolders=$wordLimit; count=$stages.Count; withinLimit=($stages.Count -le $wordLimit); folders=$stages } | ConvertTo-Json -Depth 5
  } else {
    $stage = Resolve-Stage $Path
    if ($Action -eq 'Adopt') {
      if (!(Test-Path -LiteralPath $stage -PathType Container)) { throw 'Build does not exist.' }
      if (Test-Path -LiteralPath (Join-Path $stage $wordMarker)) { [void](Read-Marker $stage) }
      else {
        $package = Get-Content -LiteralPath (Join-Path $stage 'package.json') -Raw | ConvertFrom-Json
        if ($package.name -ne 'brailleme') { throw 'Existing folder is not a BrailleMe project.' }
        Save-Json (Join-Path $stage $wordMarker) @{ project='brailleme'; repository=$wordRepo; path=$stage; createdUtc=(Get-Item -LiteralPath $stage).CreationTimeUtc.ToString('o'); adoptedUtc=[DateTime]::UtcNow.ToString('o') }
      }
      Write-Output $stage
    } elseif ($Action -eq 'Archive') { Archive-Stage $stage }
    elseif ($Action -eq 'Ensure' -and (Test-Path -LiteralPath $stage)) {
      [void](Read-Marker $stage)
      if (@(Get-Stages).Count -gt $wordLimit) { throw 'More than two build folders exist; reconcile before building.' }
      Write-Output $stage
    } else {
      if (Test-Path -LiteralPath $stage) { throw 'Destination already exists; use Ensure to reuse a registered slot.' }
      $stages = @(Get-Stages)
      foreach ($entry in $stages) { [void](Read-Marker $entry.path) }
      while ($stages.Count -ge $wordLimit) {
        $candidate = $null
        foreach ($entry in ($stages | Sort-Object createdUtc)) {
          try { Assert-Idle $entry.path; $candidate=$entry.path; break } catch { Write-Host $_.Exception.Message }
        }
        if (!$candidate) { throw 'No safely removable build folder; creation refused.' }
        Remove-ArchivedStage $candidate
        $stages = @(Get-Stages)
      }
      [void](Resolve-Stage $stage)
      New-Item -ItemType Directory -Path $stage | Out-Null
      Save-Json (Join-Path $stage $wordMarker) @{ project='brailleme'; repository=$wordRepo; path=$stage; createdUtc=[DateTime]::UtcNow.ToString('o') }
      Write-Output $stage
    }
  }
} finally { $wordLock.Dispose() }
