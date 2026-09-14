param(
    [Parameter(Mandatory = $true)] [string]$Video,
    [string]$Path = "training-video",
    [int]$Port = 8554,
    [string]$MediaMTX = "mediamtx.exe",
    [string]$FFmpeg = "ffmpeg.exe",`r`n    [int]$MinimumWidth = 512,
    [switch]$Loop,
    [switch]$KeepServer
)

$ErrorActionPreference = "Stop"
$VideoPath = (Resolve-Path -LiteralPath $Video -ErrorAction Stop).Path
if (-not (Get-Command $MediaMTX -ErrorAction SilentlyContinue)) { throw "MediaMTX was not found. Add mediamtx.exe to PATH or pass -MediaMTX." }
if (-not (Get-Command $FFmpeg -ErrorAction SilentlyContinue)) { throw "FFmpeg was not found. Add ffmpeg.exe to PATH or pass -FFmpeg." }
if ($Path -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$') { throw "-Path contains invalid characters." }
$Temp = Join-Path ([System.IO.Path]::GetTempPath()) ("scoop-ai-rtsp-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Temp | Out-Null
$Config = Join-Path $Temp "mediamtx.yml"
@"
rtspAddress: :$Port
logLevel: info
paths:
  ${Path}:
    source: publisher
"@ | Set-Content -LiteralPath $Config -Encoding ascii
$Server = $null; $Publisher = $null
try {
    $Server = Start-Process -FilePath $MediaMTX -ArgumentList $Config -PassThru
    Start-Sleep -Milliseconds 800
    $Url = "rtsp://127.0.0.1:$Port/$Path"
    $LoopArgs = if ($Loop) { @("-stream_loop", "-1") } else { @() }
    $Scale = "scale=if(gt(iw,$MinimumWidth),iw,$MinimumWidth):-2"`r`n    $Args = @("-re") + $LoopArgs + @("-i", $VideoPath, "-vf", $Scale, "-an", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-f", "rtsp", "-rtsp_transport", "tcp", $Url)
    Write-Host "Publishing $VideoPath"
    Write-Host "RTSP URL: $Url"
    Write-Host "Press Ctrl+C to stop."
    $Publisher = Start-Process -FilePath $FFmpeg -ArgumentList $Args -NoNewWindow -PassThru
    $Publisher.WaitForExit()
    if ($Publisher.ExitCode -ne 0) { throw "FFmpeg exited with code $($Publisher.ExitCode)." }
}
finally {
    if ($Publisher -and -not $Publisher.HasExited) { Stop-Process -Id $Publisher.Id -Force }
    if ($Server -and -not $Server.HasExited -and -not $KeepServer) { Stop-Process -Id $Server.Id -Force }
    Remove-Item -LiteralPath $Temp -Recurse -Force -ErrorAction SilentlyContinue
}


