$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

Set-Location -LiteralPath $ProjectRoot
& $PythonExe src\app.py @args
exit $LASTEXITCODE
