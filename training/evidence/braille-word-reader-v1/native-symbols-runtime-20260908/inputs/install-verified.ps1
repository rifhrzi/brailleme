param([Parameter(Mandatory=$true)][string]$WordIntegrityDirectory)
$ErrorActionPreference = 'Stop'
$wordAdb = 'C:\Users\rifqi\AppData\Local\Android\Sdk\platform-tools\adb.exe'
$wordReportPath = Join-Path $WordIntegrityDirectory 'report.json'
$wordInstallPath = Join-Path $WordIntegrityDirectory 'install.json'
if (Test-Path -LiteralPath $wordInstallPath) { throw 'Installation record already exists; use a new verification directory.' }
$wordReport = Get-Content -LiteralPath $wordReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($wordReport.integrity_passed -ne $true) { throw 'APK integrity did not pass.' }
$wordSourceApk = $wordReport.apk.file
$wordExpected = $wordReport.apk.sha256
if ($wordExpected -notmatch '^[a-f0-9]{64}$' -or (Get-FileHash -LiteralPath $wordSourceApk -Algorithm SHA256).Hash.ToLowerInvariant() -ne $wordExpected) { throw 'APK hash mismatch before install.' }
if ((& $wordAdb -s emulator-5554 shell getprop ro.kernel.qemu).Trim() -ne '1') { throw 'Target is not QEMU.' }
$wordAvd = (& $wordAdb -s emulator-5554 emu avd name) -join "`n"
if ($wordAvd -notmatch '^BrailleMe_API_36\s') { throw 'Wrong AVD.' }
$wordArchivedApk = Join-Path $WordIntegrityDirectory 'verified.apk'
[System.IO.File]::Copy($wordSourceApk, $wordArchivedApk, $false)
if ((Get-FileHash -LiteralPath $wordArchivedApk -Algorithm SHA256).Hash.ToLowerInvariant() -ne $wordExpected) { throw 'Archived APK copy differs.' }
$wordStarted = [DateTime]::UtcNow.ToString('o')
$wordInstall = & $wordAdb -s emulator-5554 install -r $wordArchivedApk 2>&1
$wordExit = $LASTEXITCODE
$wordRecord = @{ startedUtc=$wordStarted; completedUtc=[DateTime]::UtcNow.ToString('o'); serial='emulator-5554'; avd=$wordAvd; apk=$wordArchivedApk; sha256=$wordExpected; exitCode=$wordExit; output=($wordInstall -join "`n"); installedHashMatches=$false }
$wordRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $wordInstallPath -Encoding UTF8
$wordInstall
if ($wordExit -ne 0) { exit $wordExit }
$wordRemotePackage = (& $wordAdb -s emulator-5554 shell pm path com.brailleme.app) -join "`n"
if ($LASTEXITCODE -ne 0 -or $wordRemotePackage.Trim() -notmatch '^package:(/data/app/[A-Za-z0-9_./=+~\-]+/base\.apk)$') { throw 'Unexpected installed APK path.' }
$wordRemoteApk = $Matches[1]
$wordInstalledHash = & $wordAdb -s emulator-5554 shell sha256sum $wordRemoteApk
if ($LASTEXITCODE -ne 0 -or $wordInstalledHash -notmatch ('^' + $wordExpected + '\s')) { throw 'Installed APK hash mismatch.' }
$wordRecord.installedPath = $wordRemoteApk
$wordRecord.installedHashOutput = $wordInstalledHash
$wordRecord.installedHashMatches = $true
$wordRecord.completedUtc = [DateTime]::UtcNow.ToString('o')
$wordRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $wordInstallPath -Encoding UTF8
$wordRecord | ConvertTo-Json -Depth 5
