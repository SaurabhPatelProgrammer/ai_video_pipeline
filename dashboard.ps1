param(
    [string]$ProductRoot = "D:\ip-camera-ai-data",
    [string]$CheckpointManifest,
    [int]$Port = 8090,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Executable = Join-Path $ProjectRoot ".venv\Scripts\scoop-ai.exe"
$DashboardUrl = "http://127.0.0.1:$Port"

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Scoop AI is not installed. Run .\setup.ps1 first."
}

if ([string]::IsNullOrWhiteSpace($CheckpointManifest)) {
    $CheckpointManifest = Join-Path $ProjectRoot "models\ice-cream-item-rfdetr-nano-v2\model-manifest.json"
}
if (-not (Test-Path -LiteralPath $CheckpointManifest -PathType Leaf)) {
    throw "Approved model bundle was not found. Run git lfs pull or reinstall Scoop AI."
}

if (-not $NoBrowser) {
    try {
        Invoke-WebRequest -Uri "$DashboardUrl/api/product/status" -UseBasicParsing -TimeoutSec 1 | Out-Null
        Start-Process $DashboardUrl
        exit 0
    }
    catch {
        $HostArguments = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath,
            "-ProductRoot", $ProductRoot, "-CheckpointManifest", $CheckpointManifest,
            "-Port", $Port, "-NoBrowser"
        )
        Start-Process -FilePath (Join-Path $PSHOME "powershell.exe") -ArgumentList $HostArguments -WindowStyle Hidden
        for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
            Start-Sleep -Milliseconds 250
            try {
                Invoke-WebRequest -Uri "$DashboardUrl/api/product/status" -UseBasicParsing -TimeoutSec 1 | Out-Null
                Start-Process $DashboardUrl
                exit 0
            }
            catch { }
        }
        throw "Scoop AI dashboard did not start."
    }
}

$Arguments = @(
    "product", "--product-root", $ProductRoot,
    "--checkpoint-manifest", $CheckpointManifest,
    "--port", $Port, "--no-browser"
)

& $Executable @Arguments
exit $LASTEXITCODE
