param(
    [string]$ProductRoot = "D:\ip-camera-ai-data"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Launcher = Join-Path $ProjectRoot "dashboard.ps1"
if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
    throw "Dashboard launcher was not found at $Launcher"
}

$Desktop = [Environment]::GetFolderPath("Desktop")
if ([string]::IsNullOrWhiteSpace($Desktop)) {
    throw "Windows desktop directory could not be resolved."
}

$ShortcutPath = Join-Path $Desktop "Scoop AI Dashboard.lnk"
$PowerShell = Join-Path $PSHOME "powershell.exe"
$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $PowerShell
$Shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$Launcher`" -ProductRoot `"$ProductRoot`""
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Description = "Open the local Scoop AI operator dashboard"
$Shortcut.Save()

$Startup = [Environment]::GetFolderPath("Startup")
if (-not [string]::IsNullOrWhiteSpace($Startup)) {
    $StartupPath = Join-Path $Startup "Scoop AI Background.lnk"
    $StartupShortcut = $Shell.CreateShortcut($StartupPath)
    $StartupShortcut.TargetPath = $PowerShell
    $StartupShortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Launcher`" -ProductRoot `"$ProductRoot`" -NoBrowser"
    $StartupShortcut.WorkingDirectory = $ProjectRoot
    $StartupShortcut.Description = "Start Scoop AI local monitoring dashboard"
    $StartupShortcut.Save()
}

Write-Host "Desktop shortcut created: $ShortcutPath"
