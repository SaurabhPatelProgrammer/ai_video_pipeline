# Scoop AI production operations

The production path is a single-camera Windows edge service. Legacy OpenCV
commands remain diagnostics and experiments; they are not the canonical event
store.

## Data directories

Create and protect `D:\ip-camera-ai-data` with these subdirectories:

```text
database/   SQLite WAL event store
evidence/   event images/clips subject to retention
models/     approved checkpoint bundles and manifests
raw/        source recordings
datasets/   immutable dataset versions
logs/       rotated service logs
archive/    selected historical benchmark material
service/    Windows service settings
```

Enable BitLocker on the volume and restrict the directory ACL to the service
identity, Administrators, and the designated reviewer account.

## First setup

```powershell
.\setup.ps1
Copy-Item configs\service.example.toml configs\service.toml
Copy-Item configs\cameras\shop-01-counter-01.example.toml configs\cameras\shop-01-counter-01.toml
```

Calibrate the normalized tub and serving polygons against the locked camera
view. Do not put RTSP credentials in TOML, `.env`, command history, or service
arguments. Provision the complete RTSP URL in Windows Credential Manager:

```powershell
.\.venv\Scripts\scoop-ai.exe credential-set --key scoop-ai/shop-01-counter-01/rtsp-url --database D:\ip-camera-ai-data\database\events.sqlite3
```

Validate connectivity:

```powershell
.\.venv\Scripts\scoop-ai.exe camera-check `
  --camera-config configs\cameras\shop-01-counter-01.toml `
  --frames 30
```

Create the versioned calibration profile from the same approved camera view;
it stores an immutable reference fingerprint used to suppress inference after
camera movement or zone obstruction:

```powershell
.\.venv\Scripts\scoop-ai.exe recalibrate `
  --source <camera-or-recording> `
  --base-profile D:\ip-camera-ai-data\models\scoop-v1\profile.json `
  --output D:\ip-camera-ai-data\calibration\shop-01-counter-01.json
```

Set `camera.calibration_profile` in the camera TOML to this approved profile.
In production, the service refuses to start without it. A new mount, zoom,
resolution, or zone layout requires a new versioned profile and review.

## Dataset and model gate

Frame extraction creates an immutable version directory:

```powershell
.\extract-training-frames.ps1 video1.mp4 video2.mp4 `
  --dataset-version v1-pilot-capture --interval 0.5
```

COCO image records must retain `source_session`, `source_sha256`, and
`timestamp_seconds`. Validate before training:

```powershell
.\.venv\Scripts\scoop-ai.exe dataset validate `
  --dataset D:\ip-camera-ai-data\datasets\scoop-v1
```

Training writes `model-manifest.json` beside the selected checkpoint. The
service refuses missing files, class mismatches, path traversal, and SHA-256
mismatches.

## Silent-pilot service

Run interactively first:

```powershell
.\.venv\Scripts\scoop-ai.exe service `
  --service-config configs\service.toml `
  --camera-config configs\cameras\shop-01-counter-01.toml `
  --checkpoint-manifest D:\ip-camera-ai-data\models\scoop-v1\model-manifest.json
```

Health and metrics bind to loopback only:

```text
http://127.0.0.1:8080/health
http://127.0.0.1:8080/metrics
```

Install the same command as an auto-start Windows service from an elevated
PowerShell:

```powershell
.\install-service.ps1 `
  -ServiceConfig configs\service.toml `
  -CameraConfig configs\cameras\shop-01-counter-01.toml `
  -CheckpointManifest D:\ip-camera-ai-data\models\scoop-v1\model-manifest.json
```

## Review and evaluation

```powershell
.\.venv\Scripts\scoop-ai.exe review `
  --database D:\ip-camera-ai-data\database\events.sqlite3 `
  --evidence-root D:\ip-camera-ai-data\evidence
```

Review decisions are append-only. The event row stores the current projection;
the audit table preserves every decision. No event may affect billing during
the silent pilot.

Evaluate exported prediction and truth arrays:

```powershell
.\.venv\Scripts\scoop-ai.exe evaluate `
  --predictions predictions.json `
  --truth ground-truth.json `
  --duration 108000
```

The command exits successfully only when precision, recall and exact
per-container count accuracy are at least 95%, and wrong-container assignment
is at most 2%.

## Operational rules

- Evidence retention is 14 days; the service removes expired registered files.
- Back up the active SQLite database with `database-backup`; do not copy the
  live `.sqlite3` file directly because its WAL may contain committed data.
  Retain backup metadata alongside each generation and rehearse restore to a
  separate directory before changing the active artifact root.
- The service reconciles the evidence directory against the database at every
  startup, recovering orphaned files that a crash left unregistered and
  flagging database records whose files went missing outside its control. Run
  `scoop-ai consistency-check --database ... --evidence-root ...` on demand;
  see [Evidence/database consistency drift](RUNBOOKS.md#evidencedatabase-consistency-drift).
- Raw video stays local and face recognition is prohibited.
- A moved, obstructed or badly blurred camera suppresses inference and marks
  health degraded.
- Deploy only checksum-approved model manifests. Roll back by restoring the
  prior manifest path and restarting the service.
- Keep [`RUNBOOKS.md`](RUNBOOKS.md) with the device and rehearse camera-loss,
  invalid-credential, GPU-OOM, disk-pressure, SQLite-recovery, clock-drift,
  calibration-drift and model-rollback procedures before pilot launch.
