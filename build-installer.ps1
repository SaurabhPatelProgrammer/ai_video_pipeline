param([switch]$SkipTests)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Run .\setup.ps1 first." }
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "models\ice-cream-item-rfdetr-nano-v2\checkpoint_best_total.pth") -PathType Leaf)) {
    throw "Model checkpoint is missing. Run git lfs pull first."
}
if (-not $SkipTests) {
    & $Python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Tests failed; installer was not built." }
}
& $Python -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is required on the build machine. Install it with .\.venv\Scripts\python.exe -m pip install pyinstaller."
}
& $Python -m PyInstaller --noconfirm --clean (Join-Path $ProjectRoot "packaging\scoop-ai-client.spec")
if ($LASTEXITCODE -ne 0) { throw "Client executable build failed." }

$Inno = @(
    "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $Inno) { throw "Inno Setup 6 is required on the build machine." }
& $Inno (Join-Path $ProjectRoot "packaging\ScoopAI.iss")
if ($LASTEXITCODE -ne 0) { throw "Windows installer compilation failed." }
Write-Host "Installer created under dist\installer"
