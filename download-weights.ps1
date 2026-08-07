$ErrorActionPreference = "Stop"

$Url = "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth"
$ExpectedSha256 = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"
$CacheDirectory = Join-Path $env:USERPROFILE ".roboflow\models"
$FinalFile = Join-Path $CacheDirectory "rf-detr-nano.pth"
$PartialFile = "$FinalFile.part"

New-Item -ItemType Directory -Path $CacheDirectory -Force | Out-Null

if (Test-Path -LiteralPath $FinalFile) {
    $ExistingSha256 = (Get-FileHash -LiteralPath $FinalFile -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ExistingSha256 -eq $ExpectedSha256) {
        Write-Host "Verified RF-DETR Nano weights already exist: $FinalFile"
        exit 0
    }
    Write-Warning "Existing weight file failed checksum; a verified replacement will be downloaded."
}

Write-Host "Downloading official RF-DETR Nano weights with resume support..."
Write-Host "If the network stops, run this script again; the .part file will resume."

& curl.exe `
    --location `
    --fail `
    --continue-at - `
    --retry 20 `
    --retry-delay 5 `
    --retry-all-errors `
    --connect-timeout 30 `
    --output $PartialFile `
    $Url

if ($LASTEXITCODE -ne 0) {
    throw "Download did not complete. Keep the .part file and run this script again."
}

$ActualSha256 = (Get-FileHash -LiteralPath $PartialFile -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualSha256 -ne $ExpectedSha256) {
    throw "Checksum mismatch. Expected $ExpectedSha256 but received $ActualSha256. File was not activated."
}

Move-Item -LiteralPath $PartialFile -Destination $FinalFile -Force
Write-Host "Downloaded and verified: $FinalFile"
