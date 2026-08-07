param(
    [string]$ServiceConfig,
    [string]$CameraConfig,
    [string]$CheckpointManifest,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

if ($Remove) {
    & $PythonExe -m scoop_ai.windows_service stop
    & $PythonExe -m scoop_ai.windows_service remove
    exit $LASTEXITCODE
}

foreach ($Path in @($ServiceConfig, $CameraConfig, $CheckpointManifest)) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "ServiceConfig, CameraConfig, and CheckpointManifest must be readable files."
    }
}

$SettingsDirectory = "D:\ip-camera-ai-data\service"
$SettingsPath = Join-Path $SettingsDirectory "windows-service.json"
New-Item -ItemType Directory -Path $SettingsDirectory -Force | Out-Null
@{
    service_config = (Resolve-Path -LiteralPath $ServiceConfig).Path
    camera_config = (Resolve-Path -LiteralPath $CameraConfig).Path
    checkpoint_manifest = (Resolve-Path -LiteralPath $CheckpointManifest).Path
} | ConvertTo-Json | Set-Content -LiteralPath $SettingsPath -Encoding UTF8

& $PythonExe -m scoop_ai.windows_service --startup auto install
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExe -m scoop_ai.windows_service start
exit $LASTEXITCODE
