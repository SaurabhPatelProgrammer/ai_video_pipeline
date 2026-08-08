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

# Restart only after an actual crash, with bounded delays so a bad deployment
# cannot thrash the host. The one-day reset window limits repeated restarts.
& sc.exe failure ScoopAIEdge actions= restart/60000/restart/120000/restart/300000 reset= 86400
if ($LASTEXITCODE -ne 0) { throw "Could not configure SCM crash recovery actions." }
& sc.exe failureflag ScoopAIEdge 1
if ($LASTEXITCODE -ne 0) { throw "Could not enable SCM failure actions." }

& $PythonExe -m scoop_ai.windows_service start
exit $LASTEXITCODE
