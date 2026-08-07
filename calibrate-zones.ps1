param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [string]$BaseProfile = "models\scoop-two-video-motion-v5\profile.json",
    [string]$Output = "configs\level1_shop_camera.user.json",
    [double]$SeekSeconds = 0.0,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScoopAI = Join-Path $ProjectRoot ".venv\Scripts\scoop-ai.exe"

if (-not (Test-Path -LiteralPath $ScoopAI -PathType Leaf)) {
    throw "Scoop AI CLI not found. Run .\setup.ps1 first."
}

Set-Location -LiteralPath $ProjectRoot
$Arguments = @(
    "zone-calibrate",
    "--source", $Source,
    "--base-profile", $BaseProfile,
    "--output", $Output,
    "--seek-seconds", $SeekSeconds
)
if ($Force) {
    $Arguments += "--force"
}
& $ScoopAI @Arguments
exit $LASTEXITCODE
