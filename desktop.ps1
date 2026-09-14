param(
    [string]$ProductRoot = "D:\ip-camera-ai-data",
    [string]$CheckpointManifest,
    [int]$Port = 8090,
    [switch]$SoftwareRendering,
    [switch]$StartHidden
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Executable = Join-Path $ProjectRoot ".venv\Scripts\scoop-ai.exe"

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Scoop AI is not installed. Run .\setup.ps1 first."
}

if ([string]::IsNullOrWhiteSpace($CheckpointManifest)) {
    $CheckpointManifest = Join-Path $ProjectRoot "models\ice-cream-item-rfdetr-nano-v2\model-manifest.json"
}
if (-not (Test-Path -LiteralPath $CheckpointManifest -PathType Leaf)) {
    throw "Approved model bundle was not found. Run git lfs pull or reinstall Scoop AI."
}

$Arguments = @(
    "desktop",
    "--product-root", $ProductRoot,
    "--checkpoint-manifest", $CheckpointManifest,
    "--port", $Port
)
if ($SoftwareRendering) { $Arguments += "--software-rendering" }
if ($StartHidden) { $Arguments += "--start-hidden" }

& $Executable @Arguments
exit $LASTEXITCODE
