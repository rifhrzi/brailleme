param(
  [ValidateSet('assembleRelease', ':braille-word-geometry:assembleRelease')]
  [string]$WordBuildTask = 'assembleRelease'
)
$ErrorActionPreference = 'Stop'
$wordStage = 'C:\bm-symbols-apk-20260908'
$wordJdk = 'C:\Users\rifqi\.jdks\brailleme-temurin-17\jdk-17.0.20+8'
$wordSdk = 'C:\Users\rifqi\AppData\Local\Android\Sdk'
$wordStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$wordBuildEvidence = Join-Path $wordStage ('evidence\build-' + $wordStamp)
New-Item -ItemType Directory -Path $wordBuildEvidence | Out-Null
$wordInit = Join-Path $wordBuildEvidence 'limit-native-jobs.gradle'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '..\..\training\evidence\braille-word-reader-v1\native-camera-capture-diagnostic-v3\harness\limit-native-jobs.gradle') -Destination $wordInit
$env:JAVA_HOME = $wordJdk
$env:ANDROID_HOME = $wordSdk
$env:ANDROID_SDK_ROOT = $wordSdk
$env:APP_ENV = 'preview'
$env:CI = '1'
$env:DEBUG = ''
$env:PATH = (Join-Path $wordJdk 'bin') + ';' + $env:PATH
$wordBuildArgs = @($WordBuildTask, '--no-daemon', '--no-parallel', '--max-workers=1', '--console=plain', '-PreactNativeArchitectures=arm64-v8a,x86_64', '-Pkotlin.compiler.execution.strategy=in-process', '--init-script', $wordInit)
$wordRecord = @{ startedUtc = [DateTime]::UtcNow.ToString('o'); stage = $wordStage; javaHome = $wordJdk; sdk = $wordSdk; appEnvironment = 'preview'; command = @('gradlew.bat') + $wordBuildArgs }
$wordRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $wordBuildEvidence 'build.json') -Encoding UTF8
Push-Location (Join-Path $wordStage 'android')
try {
  $ErrorActionPreference = 'Continue'
  & .\gradlew.bat @wordBuildArgs 2>&1 | Tee-Object -FilePath (Join-Path $wordBuildEvidence 'gradle.log')
  $wordExit = $LASTEXITCODE
} finally {
  $ErrorActionPreference = 'Stop'
  Pop-Location
}
$wordRecord.completedUtc = [DateTime]::UtcNow.ToString('o')
$wordRecord.exitCode = $wordExit
$wordRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $wordBuildEvidence 'build.json') -Encoding UTF8
Write-Output ('Build evidence: ' + $wordBuildEvidence)
exit $wordExit
