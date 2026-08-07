param(
    [string]$Profile = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$UserProfile = Join-Path $ProjectRoot "configs\level1_shop_camera.user.json"
$CalibratedModel = Join-Path $ProjectRoot "models\scoop-two-video-motion-v7\profile.json"
$ServedOrderModel = Join-Path $ProjectRoot "models\served-order-two-video-v8\profile.json"
$TrainedProfile = Join-Path $ProjectRoot "models\scoop-two-video-motion-v5\profile.json"
$ProfilePath = if (-not [string]::IsNullOrWhiteSpace($Profile)) {
    Join-Path $ProjectRoot $Profile
}
elseif (Test-Path -LiteralPath $ServedOrderModel -PathType Leaf) {
    $ServedOrderModel
}
elseif (Test-Path -LiteralPath $CalibratedModel -PathType Leaf) {
    $CalibratedModel
}
elseif (Test-Path -LiteralPath $UserProfile -PathType Leaf) {
    $UserProfile
}
else {
    $TrainedProfile
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}
if (-not (Test-Path -LiteralPath $ProfilePath -PathType Leaf)) {
    throw "Trained profile not found: $ProfilePath"
}

Set-Location -LiteralPath $ProjectRoot
& $PythonExe src\level1_counter.py --profile $ProfilePath @args
exit $LASTEXITCODE
