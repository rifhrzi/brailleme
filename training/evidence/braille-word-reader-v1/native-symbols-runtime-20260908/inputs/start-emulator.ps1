param(
  [string]$WordCameraImage = 'C:\bm-symbols-apk-20260908\evidence\camera-word.png'
)
$ErrorActionPreference = 'Stop'
$wordSdk = 'C:\Users\rifqi\AppData\Local\Android\Sdk'
$wordAdb = Join-Path $wordSdk 'platform-tools\adb.exe'
$wordEmulator = Join-Path $wordSdk 'emulator\emulator.exe'
$wordImage = (Resolve-Path -LiteralPath $WordCameraImage).Path
$wordDevices = & $wordAdb devices
if ($wordDevices -match '^emulator-5554\s') { throw 'Emulator5554 already exists; inspect ownership before changing it.' }
$wordStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$wordEvidence = Join-Path 'C:\bm-symbols-apk-20260908\evidence' ('emulator-' + $wordStamp)
New-Item -ItemType Directory -Path $wordEvidence | Out-Null
$wordArgs = @('-avd','BrailleMe_API_36','-port','5554','-read-only','-no-window','-no-audio','-no-snapshot','-no-boot-anim','-memory','2560','-cores','2','-gpu','swiftshader_indirect','-feature','-Vulkan','-camera-back', ('imagefile:' + $wordImage), '-camera-front','none')
$wordProcess = Start-Process -FilePath $wordEmulator -ArgumentList $wordArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $wordEvidence 'stdout.log') -RedirectStandardError (Join-Path $wordEvidence 'stderr.log')
@{ startedUtc = [DateTime]::UtcNow.ToString('o'); pid = $wordProcess.Id; serial = 'emulator-5554'; avd = 'BrailleMe_API_36'; arguments = $wordArgs; cameraImage = $wordImage; cameraImageSha256 = (Get-FileHash -LiteralPath $wordImage -Algorithm SHA256).Hash.ToLowerInvariant(); apkInstalled = $false } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $wordEvidence 'launch.json') -Encoding UTF8
Write-Output ('Emulator launched. Verify boot/AVD/QEMU before install. Evidence: ' + $wordEvidence)
