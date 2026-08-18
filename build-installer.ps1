param(
    [switch]$SkipTests,
    [switch]$SkipClientBuild,
    [switch]$Release,
    [string]$CertificateThumbprint,
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [switch]$CommercialLicenseAcknowledged,
    [long]$MaximumReleaseInstallerBytes = 2000000000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Installer = Join-Path $ProjectRoot "dist\installer\ScoopAI-Setup-0.1.0.exe"
$ClientExe = Join-Path $ProjectRoot "dist\ScoopAIClient\ScoopAIClient.exe"
$Manifest = Join-Path $ProjectRoot "dist\installer\release-manifest.json"

function Find-SignTool {
    $roots = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10\bin",
        "$env:ProgramFiles\Windows Kits\10\bin"
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Container }
    foreach ($root in $roots) {
        $candidate = Get-ChildItem -LiteralPath $root -Filter signtool.exe -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\x64\\signtool\.exe$" } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($candidate) { return $candidate.FullName }
    }
    throw "SignTool was not found. Install the Windows SDK signing tools."
}

function Sign-Artifact([string]$Path, [string]$SignTool) {
    & $SignTool sign /sha1 $CertificateThumbprint /fd SHA256 /tr $TimestampUrl /td SHA256 $Path
    if ($LASTEXITCODE -ne 0) { throw "Authenticode signing failed: $Path" }
    & $SignTool verify /pa /all $Path
    if ($LASTEXITCODE -ne 0) { throw "Authenticode verification failed: $Path" }
}

if ($Release) {
    if ($SkipClientBuild) { throw "Release builds cannot use -SkipClientBuild." }
    $dirty = & git -C $ProjectRoot status --porcelain --untracked-files=all
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect Git release state." }
    if ($dirty) { throw "Release builds require a clean Git working tree." }
    $tag = & git -C $ProjectRoot tag --points-at HEAD | Select-Object -First 1
    if ($LASTEXITCODE -ne 0 -or -not $tag) { throw "Release builds require HEAD to have an exact version tag." }
    if (-not $CommercialLicenseAcknowledged) {
        throw "Release builds require -CommercialLicenseAcknowledged for the commercial installer tool."
    }
    if (-not $CertificateThumbprint.Trim()) {
        throw "Release builds require -CertificateThumbprint for Authenticode signing."
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Run .\setup.ps1 first." }
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "models\ice-cream-item-rfdetr-nano-v2\checkpoint_best_total.pth") -PathType Leaf)) {
    throw "Model checkpoint is missing. Run git lfs pull first."
}
if (-not $SkipClientBuild) {
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
} elseif (-not (Test-Path -LiteralPath $ClientExe -PathType Leaf)) {
    throw "Existing client executable was not found for -SkipClientBuild."
}

$SignTool = $null
if ($CertificateThumbprint.Trim()) {
    $SignTool = Find-SignTool
    Sign-Artifact $ClientExe $SignTool
}

$Inno = @(
    "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $Inno) { throw "Inno Setup 6 is required on the build machine." }
& $Inno (Join-Path $ProjectRoot "packaging\ScoopAI.iss")
if ($LASTEXITCODE -ne 0) { throw "Windows installer compilation failed." }
if ($SignTool) { Sign-Artifact $Installer $SignTool }

$installerItem = Get-Item -LiteralPath $Installer
if ($Release -and $installerItem.Length -gt $MaximumReleaseInstallerBytes) {
    throw "Release installer is $($installerItem.Length) bytes; limit is $MaximumReleaseInstallerBytes. Split or reduce the bundle."
}
$commit = (& git -C $ProjectRoot rev-parse HEAD).Trim()
$releaseTag = & git -C $ProjectRoot tag --points-at HEAD | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or -not $releaseTag) { $releaseTag = $null }
$workingTreeDirty = [bool](& git -C $ProjectRoot status --porcelain --untracked-files=all)
$clientSignature = Get-AuthenticodeSignature -LiteralPath $ClientExe
$installerSignature = Get-AuthenticodeSignature -LiteralPath $Installer
$provenance = [ordered]@{
    schema_version = 1
    built_at_utc = [DateTime]::UtcNow.ToString("o")
    release = [bool]$Release
    git_commit = $commit
    git_tag = $releaseTag
    working_tree_dirty = $workingTreeDirty
    client = [ordered]@{
        path = "dist/ScoopAIClient/ScoopAIClient.exe"
        sha256 = (Get-FileHash -LiteralPath $ClientExe -Algorithm SHA256).Hash.ToLowerInvariant()
        size_bytes = (Get-Item -LiteralPath $ClientExe).Length
        signature_status = [string]$clientSignature.Status
    }
    installer = [ordered]@{
        path = "dist/installer/ScoopAI-Setup-0.1.0.exe"
        sha256 = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
        size_bytes = $installerItem.Length
        signature_status = [string]$installerSignature.Status
    }
}
$provenance | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Manifest -Encoding UTF8
Write-Host "Installer and release manifest created under dist\installer"
