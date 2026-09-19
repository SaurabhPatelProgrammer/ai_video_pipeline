param(
    [ValidateSet("auto", "cu130", "cu128", "cpu")]
    [string]$Compute = "auto",
    [switch]$SkipDashboardShortcut,
    [switch]$IncludeDevTools
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

if ($Compute -eq "auto") {
    # Installing a CUDA build on a machine with no NVIDIA driver downloads
    # gigabytes that can never be used, so probe the driver before choosing.
    $HasNvidia = $false
    try {
        $null = & nvidia-smi -L 2>$null
        $HasNvidia = ($LASTEXITCODE -eq 0)
    }
    catch {
        $HasNvidia = $false
    }
    if ($HasNvidia) {
        $Compute = "cu130"
        Write-Host "NVIDIA GPU detected. Installing the CUDA 13.0 runtime."
    }
    else {
        $Compute = "cpu"
        Write-Host "No NVIDIA GPU detected. Installing the CPU runtime; inference will be slower."
    }
}

$Python311 = $null
try {
    $LauncherPython = py -3.11 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $LauncherPython)) {
        $Python311 = $LauncherPython
    }
}
catch {
    $Python311 = $null
}

if ($null -eq $Python311) {
    $UserPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
    if (Test-Path -LiteralPath $UserPython) {
        $Python311 = $UserPython
    }
}

if ($null -eq $Python311) {
    throw "Python 3.11 is required but was not found. Install it, then run this script again."
}

if (-not (Test-Path -LiteralPath ".venv")) {
    & $Python311 -m venv .venv
}

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $PythonExe -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Python packaging tools could not be installed." }

if ($Compute -eq "cu130") {
    & $PythonExe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
}
elseif ($Compute -eq "cu128") {
    & $PythonExe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
}
else {
    & $PythonExe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw "PyTorch could not be installed for compute target $Compute." }

& $PythonExe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Project dependencies could not be installed." }
if ($IncludeDevTools) {
    & $PythonExe -m pip install -e ".[dev]"
}
else {
    & $PythonExe -m pip install --no-deps -e .
}
if ($LASTEXITCODE -ne 0) { throw "The scoop-ai package could not be installed." }

& $PythonExe scripts\check_environment.py
if ($LASTEXITCODE -ne 0) { throw "Environment verification failed." }
if (-not $SkipDashboardShortcut) {
    & (Join-Path $ProjectRoot "install-dashboard-shortcut.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Dashboard shortcut could not be created." }
}
& $PythonExe -m scoop_ai.cli compute-check
Write-Host ""
Write-Host "Setup complete. Use the Scoop AI desktop shortcut after provisioning the camera."
if ($IncludeDevTools) {
    Write-Host "Developer tools are installed. Run '$PythonExe -m pytest -q' to validate the project."
}
Write-Host "Provision the camera with 'scoop-ai credential-set', then validate it with 'scoop-ai camera-check'."
