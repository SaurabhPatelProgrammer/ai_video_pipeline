# IP Camera AI MVP

Fixed-camera video analytics for detecting completed ice-cream serving events.
The repository contains a production-oriented Windows edge-service foundation,
diagnostic OpenCV tools, dataset validation, model-manifest verification,
review workflows, and experimental motion baselines.

The current motion profiles are review-only prototypes. They are not billing-
grade counters and must not drive autonomous commercial decisions. The canonical
production path is the installable `scoop_ai` package documented in
[`docs/PRODUCTION.md`](docs/PRODUCTION.md).

## Repository status

The implemented service pipeline is:

```text
camera or recorded video
        -> capture and timestamping
        -> frame-quality gate
        -> RF-DETR detector
        -> ByteTrack object tracker
        -> release-aware deposit state machine
        -> SQLite event store and evidence files
        -> local review application
```

The service currently emits reviewed-pilot candidate events. Production
approval requires the acceptance gates recorded with each deployment.

## Privacy and repository boundaries

The repository intentionally contains source code, tests, documentation,
PowerShell launchers, and safe example configuration only. The following are
local artifacts and are excluded by `.gitignore`:

- `.env`, camera credentials, and site-specific camera configuration;
- raw video, extracted frames, evidence, reports, and databases;
- trained profiles, checkpoints, downloaded weights, and virtual environments.

Before sharing logs or reports, verify that they do not contain camera URLs,
credentials, customer or employee imagery, network addresses, or private local
paths.

## Requirements

- Windows 10 or 11;
- Python 3.11 (the project requires `>=3.11,<3.12`);
- an NVIDIA driver and GPU for practical model inference;
- a reachable webcam, RTSP camera, HTTP stream, or recorded video file.

## Installation

```powershell
git clone git@github.com:SaurabhPatelProgrammer/ai_video_pipeline.git
Set-Location ai_video_pipeline
Copy-Item .env.example .env
.\setup.ps1
```

The setup script installs the selected PyTorch build, project dependencies, and
the editable `scoop-ai` package. CUDA 13.0 is the default; use `-Compute cu128`
or `-Compute cpu` when appropriate.

## Production camera credentials

Do not place RTSP credentials in `.env`, TOML, command-line arguments, shell
history, screenshots, or source control. Provision the complete source URL in
Windows Credential Manager using the same hierarchical key referenced by the
example camera configuration:

```powershell
.\.venv\Scripts\scoop-ai.exe credential-set `
  --key scoop-ai/shop-01-counter-01/rtsp-url
```

Copy and customize the example configurations:

```powershell
Copy-Item configs\service.example.toml configs\service.toml
Copy-Item configs\cameras\shop-01-counter-01.example.toml `
  configs\cameras\shop-01-counter-01.toml
```

Calibrate the normalized tub and serving polygons against the locked camera
view. The complete deployment procedure is in
[`docs/PRODUCTION.md`](docs/PRODUCTION.md).

## Camera diagnostics

Run a camera check before loading an AI model:

```powershell
.\check-camera.ps1
.\check-camera.ps1 --headless --seconds 10
```

For a production camera, use the credential-backed configuration through the
CLI:

```powershell
.\.venv\Scripts\scoop-ai.exe camera-check `
  --camera-config configs\cameras\shop-01-counter-01.toml `
  --frames 30
```

Use an H.264 sub-stream for the initial RTSP test. If frames are black or
unstable, verify the vendor-specific RTSP path, codec, firewall, and camera
network reachability.

## RF-DETR preview

The legacy preview path validates general object detection and approximate
latency. A pretrained COCO model does not understand a completed ice-cream
deposit; shop-specific detection requires labeled data and fine-tuning.

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
.\download-weights.ps1
.\run.ps1
```

Useful examples:

```powershell
.\run.ps1 --model small --confidence 0.45
.\run.ps1 --classes "person,cup,bowl,bottle"
.\run.ps1 --source "sample.mp4" --seconds 30
```

The display controls are `Q` or `Esc` to exit and `S` to save a snapshot.
Start with the `nano` model on an RTX 3050-class GPU and benchmark larger
models only after the camera pipeline is stable.

## Legacy object-crossing counter

The simple counter detects recognized COCO objects crossing a virtual line. It
is a diagnostic utility, not a scoop-event detector:

```powershell
.\count-objects.ps1
.\count-objects.ps1 --classes "person,cup,bottle"
.\count-objects.ps1 --line "0,540,1919,540"
```

An empty class filter means every class recognized by the pretrained model, not
every physical object in the scene.

## Development workbench

The workbench supports recorded-video inspection, stable tracking IDs, manual
ground truth, evidence snapshots, and candidate-event review:

```powershell
.\workbench.ps1
.\workbench.ps1 --source "sample.mp4"
```

Without a custom checkpoint it runs in `COCO PREVIEW` mode. Generic detections
may be visible, but scoop-event counting remains disabled. With a trained
checkpoint, set `SCOOP_CHECKPOINT` in the local `.env` file.

Controls:

- `G`: record one true completed scoop;
- `U`: undo the latest manual count;
- `R`: reset the current scenario;
- `S`: save a diagnostic snapshot;
- `Q` or `Esc`: finish the session.

Session artifacts are written under `artifacts/workbench/<timestamp>` and are
ignored by Git.

## Dataset and model workflow

Extract traceable frames from source videos:

```powershell
.\extract-training-frames.ps1 video1.mp4 video2.mp4 --interval 0.5
```

Each COCO split must contain an `_annotations.coco.json` file. Keep complete
recording sessions isolated between train, validation, and test splits; random
frame-level splitting creates leakage and overstates model quality.

Validate the dataset before training:

```powershell
.\.venv\Scripts\scoop-ai.exe dataset validate `
  --dataset D:\ip-camera-ai-data\datasets\scoop-v1
```

The validation layer checks source-session leakage, duplicate image content,
category consistency, bounding-box bounds, and reproducible fingerprints.
Model manifests verify checkpoint location, architecture, class names, and
SHA-256 before inference loads a checkpoint.

## Level-1 motion baseline

The Level-1 baseline uses motion in calibrated tub/load and serving zones to
generate candidate events from limited unlabelled footage:

```powershell
.\run-level1.ps1 --source "sample.mp4"
.\run-level1.ps1 --playback-speed 2
```

It does not recognize a loaded scoop visually and does not assign individual
cups or cones. Every candidate must be compared with manual ground truth.

## Testing

The test suite is written with the standard `unittest` runner and is also
configured for pytest in CI:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src tests scripts
```

## Documentation map

- [`docs/PRODUCTION.md`](docs/PRODUCTION.md): secure setup and service operation;
- [`docs/RUNBOOKS.md`](docs/RUNBOOKS.md): operational incident procedures.
