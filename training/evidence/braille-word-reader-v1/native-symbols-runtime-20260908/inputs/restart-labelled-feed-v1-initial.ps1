param([Parameter(Mandatory=$true)][ValidateSet('lowercase-abc.png','numbers-2026.png','numbers-001.png')][string]$Feed)
$ErrorActionPreference='Stop'
$wordAdb='C:\Users\rifqi\AppData\Local\Android\Sdk\platform-tools\adb.exe'
$wordBase='C:\bm-symbols-apk-20260908\evidence'
$wordManifest=Join-Path $wordBase 'calibrated-feeds-v1\manifest.json'
$wordImage=Join-Path (Split-Path $wordManifest) $Feed
$wordRow=(Get-Content -LiteralPath $wordManifest -Raw -Encoding UTF8 | ConvertFrom-Json).rows | Where-Object output -eq $Feed
if (@($wordRow).Count -ne 1 -or (Get-FileHash -LiteralPath $wordImage -Algorithm SHA256).Hash.ToLowerInvariant() -ne $wordRow.output_sha256) { throw 'Frozen feed identity mismatch' }
$wordApk=Join-Path $PSScriptRoot '..\android\BrailleMe-0.3.0-symbols-auto-frame-2026-09-09.apk'
$wordExpected='9afa6a1eb2983d0a92b4f1dcff3026f91d583bca02cda6d98cfaf91760a90259'
if ((Get-FileHash -LiteralPath $wordApk -Algorithm SHA256).Hash.ToLowerInvariant() -ne $wordExpected) { throw 'APK identity mismatch' }
$wordOut=Join-Path $wordBase ('feed-' + [IO.Path]::GetFileNameWithoutExtension($Feed) + '-v6')
if (Test-Path -LiteralPath $wordOut) { throw 'Use a new evidence destination' }
New-Item -ItemType Directory -Path $wordOut | Out-Null
if ((& $wordAdb -s emulator-5554 shell getprop ro.kernel.qemu).Trim() -ne '1') { throw 'Not QEMU' }
$wordAvd=(& $wordAdb -s emulator-5554 emu avd name) -join "`n"
if ($wordAvd -notmatch '^BrailleMe_API_36\s') { throw 'Wrong owned AVD' }
& $wordAdb -s emulator-5554 logcat -d -b events -v epoch | Set-Content -LiteralPath (Join-Path $wordOut 'prior-emulator-events.log') -Encoding UTF8
& $wordAdb -s emulator-5554 emu kill | Set-Content -LiteralPath (Join-Path $wordOut 'stop-owned-emulator.log') -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw 'Could not stop owned emulator' }
$wordDeadline=[DateTime]::UtcNow.AddSeconds(30)
do { Start-Sleep -Seconds 1; $wordDevices=& $wordAdb devices } while (($wordDevices -match '^emulator-5554\s') -and [DateTime]::UtcNow -lt $wordDeadline)
if ($wordDevices -match '^emulator-5554\s') { throw 'Owned emulator still present' }
& (Join-Path $PSScriptRoot 'start-emulator.ps1') -WordCameraImage $wordImage
$wordDeadline=[DateTime]::UtcNow.AddSeconds(180)
do { Start-Sleep -Seconds 2; $wordBoot=& $wordAdb -s emulator-5554 shell getprop sys.boot_completed 2>$null } while ($wordBoot -ne '1' -and [DateTime]::UtcNow -lt $wordDeadline)
if ($wordBoot -ne '1') { throw 'Boot deadline exceeded' }
if ((& $wordAdb -s emulator-5554 shell getprop ro.kernel.qemu).Trim() -ne '1') { throw 'Booted target not QEMU' }
$wordAvd=(& $wordAdb -s emulator-5554 emu avd name) -join "`n"
if ($wordAvd -notmatch '^BrailleMe_API_36\s') { throw 'Booted wrong AVD' }
$wordInstall=& $wordAdb -s emulator-5554 install -r $wordApk 2>&1
$wordInstall | Set-Content -LiteralPath (Join-Path $wordOut 'install.log') -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw 'APK install failed' }
$wordPackage=(& $wordAdb -s emulator-5554 shell pm path com.brailleme.app) -join "`n"
if ($wordPackage.Trim() -notmatch '^package:(/data/app/[A-Za-z0-9_./=+~\-]+/base\.apk)$') { throw 'Unexpected installed APK path' }
$wordRemote=$Matches[1]
$wordHash=& $wordAdb -s emulator-5554 shell sha256sum $wordRemote
if ($wordHash -notmatch ('^' + $wordExpected + '\s')) { throw 'Installed APK mismatch' }
$wordDex=Join-Path $wordBase 'ui-dump-no-idle-build-v3\classes.dex'
$wordDexExpected='79e234d57492167293889ef7df22815d53caa2f376b8e1ccaf1da031f4dee30a'
if ((Get-FileHash -LiteralPath $wordDex -Algorithm SHA256).Hash.ToLowerInvariant() -ne $wordDexExpected) { throw 'UI helper mismatch' }
& $wordAdb -s emulator-5554 push $wordDex /data/local/tmp/brailleme-ui-dump-v3.dex
if ($LASTEXITCODE -ne 0) { throw 'Could not stage UI helper' }
& $wordAdb -s emulator-5554 shell input keyevent 224
& $wordAdb -s emulator-5554 shell wm dismiss-keyguard
& $wordAdb -s emulator-5554 logcat -d -b events -v epoch | Set-Content -LiteralPath (Join-Path $wordOut 'boot-events.log') -Encoding UTF8
@{ utc=[DateTime]::UtcNow.ToString('o'); feed=$Feed; feedSha256=$wordRow.output_sha256; manifestSha256=(Get-FileHash -LiteralPath $wordManifest -Algorithm SHA256).Hash.ToLowerInvariant(); apkSha256=$wordExpected; installedApk=$wordRemote; installedHashOutput=$wordHash; installedHashMatches=$true; helperSha256=$wordDexExpected; scriptSha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant(); cameraVisualVerification='pending'; avd=$wordAvd } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $wordOut 'installation.json') -Encoding UTF8
Write-Output ('Feed restarted and v6 installed: ' + $wordOut)
