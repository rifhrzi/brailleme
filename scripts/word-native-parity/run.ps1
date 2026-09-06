param(
  [string]$AndroidSdk = "$env:LOCALAPPDATA\Android\Sdk",
  [Parameter(Mandatory = $true)][string]$JavaDirectory,
  [string]$StagingDirectory = 'C:\bm-word-native-parity-20260905',
  [string]$Serial,
  [switch]$CompileOnly
)
$ErrorActionPreference = 'Stop'
$stage = [System.IO.Path]::GetFullPath($StagingDirectory)
if (-not (Test-Path -LiteralPath $stage)) { New-Item -ItemType Directory -Path $stage | Out-Null }
$platform = Join-Path $AndroidSdk 'platforms\android-36\android.jar'
$dexCompiler = Join-Path $AndroidSdk 'build-tools\36.0.0\lib\d8.jar'
$javac = Join-Path $JavaDirectory 'bin\javac.exe'
$java = Join-Path $JavaDirectory 'bin\java.exe'
$adb = Join-Path $AndroidSdk 'platform-tools\adb.exe'
$source = Join-Path $PSScriptRoot 'WordNativeParity.java'
& node (Join-Path $PSScriptRoot 'geometry-reference.cjs') (Join-Path $stage 'geometry-reference.json')
if ($LASTEXITCODE -ne 0) { throw 'TypeScript geometry reference failed.' }
& $javac --release 8 -cp $platform -d $stage $source
if ($LASTEXITCODE -ne 0) { throw 'Java compilation failed.' }
& $java -cp $dexCompiler com.android.tools.r8.D8 --lib $platform --min-api 26 --output $stage (Join-Path $stage 'WordNativeParity.class')
if ($LASTEXITCODE -ne 0) { throw 'DEX compilation failed.' }
if ($CompileOnly) { Write-Output "Compiled harness: $stage"; exit 0 }
$deviceArgs = if ($Serial) { @('-s', $Serial) } else { @() }
$remote = '/data/local/tmp/braille-word-parity-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& $adb @deviceArgs shell mkdir -p $remote
if ($LASTEXITCODE -ne 0) { throw 'No usable Android device. Start an emulator or specify -Serial.' }
& $adb @deviceArgs push (Join-Path $stage 'classes.dex') "$remote/classes.dex"
if ($LASTEXITCODE -ne 0) { throw 'Cannot copy the DEX harness.' }
& $adb @deviceArgs push (Join-Path $stage 'geometry-reference.json') "$remote/geometry-reference.json"
if ($LASTEXITCODE -ne 0) { throw 'Cannot copy the TypeScript coordinates.' }
& $adb @deviceArgs shell chmod 444 "$remote/classes.dex"
& $adb @deviceArgs shell "CLASSPATH=$remote/classes.dex" app_process /system/bin WordNativeParity "$remote/output" "$remote/geometry-reference.json"
if ($LASTEXITCODE -ne 0) { throw 'Android Bitmap harness failed.' }
& $adb @deviceArgs pull "$remote/output" $stage
if ($LASTEXITCODE -ne 0) { throw 'Cannot pull native evidence.' }
Write-Output "Native evidence: $(Join-Path $stage 'output')"
