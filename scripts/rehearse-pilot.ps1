param(
    [Parameter(Mandatory = $true)][string]$Database,
    [Parameter(Mandatory = $true)][string]$BackupDirectory,
    [Parameter(Mandatory = $true)][string]$RestoreDirectory,
    [string]$CameraId,
    [string]$ApprovedReviewer
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}
if (-not (Test-Path -LiteralPath $Database -PathType Leaf)) {
    throw "Database does not exist: $Database"
}

New-Item -ItemType Directory -Path $BackupDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $RestoreDirectory -Force | Out-Null
& $PythonExe -m scoop_ai.cli database-backup --database $Database --output-dir $BackupDirectory
if ($LASTEXITCODE -ne 0) { throw "Database backup rehearsal failed." }
$backup = Get-ChildItem -LiteralPath $BackupDirectory -Filter *.sqlite3 | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $backup) { throw "No backup artifact was produced." }
& $PythonExe -m scoop_ai.cli database-restore --backup $backup.FullName --output-dir $RestoreDirectory
if ($LASTEXITCODE -ne 0) { throw "Database restore rehearsal failed." }
$restored = Join-Path $RestoreDirectory "events.sqlite3"
& $PythonExe -m scoop_ai.cli database-check --database $restored
if ($LASTEXITCODE -ne 0) { throw "Restored database integrity check failed." }
if ($CameraId -and $ApprovedReviewer) {
    & $PythonExe -m scoop_ai.cli model-rollback --database $restored --camera-id $CameraId --approved-reviewer $ApprovedReviewer
    if ($LASTEXITCODE -ne 0) { throw "Rollback rehearsal failed on the restored copy." }
}
Write-Output "Pilot backup, restore, integrity, and optional rollback rehearsal passed."
