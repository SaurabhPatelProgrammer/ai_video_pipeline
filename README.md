# IP Camera AI MVP

> नया user शुरू कर रहा है या पूरे project को non-technical भाषा में समझना है? पहले **[संपूर्ण Hindi project guide](docs/COMPLETE-PROJECT-GUIDE-HINDI.md)** पढ़ें। इसमें installation, final model replay, zones, training, production operations, troubleshooting और file-by-file reference शामिल हैं।

Fixed-camera video analytics for detecting completed ice-cream serving events.
The repository contains a production-oriented Windows edge-service foundation,
diagnostic OpenCV tools, dataset validation, model-manifest verification,
review workflows, and experimental motion baselines.

The current motion profiles are review-only prototypes. They are not billing-
grade counters and must not drive autonomous commercial decisions. The canonical
production path is the installable `scoop_ai` package documented in
[`docs/PRODUCTION.md`](docs/PRODUCTION.md).

## Repository status

The implemented canonical service selects a model-compatible event pipeline:

```text
camera or recorded video
        -> capture and timestamping
        -> frame-quality gate
        -> RF-DETR detector
        -> one-class proximity tracking -> ice-cream handover state machine
           OR three-class ByteTrack -> release-aware deposit state machine
        -> SQLite event store and evidence files
        -> local review application
```

The committed V2 model contains the single `ice_cream_item` class, so it uses
the handover pipeline. Set `camera.pipeline = "handover"` in production camera
configuration; `auto` is available for development and selects from the
manifest classes. The older three-class deposit path remains supported only
for manifests containing exactly `scoop`, `loaded_scoop`, and
`serving_container`.

The service currently emits reviewed-pilot candidate events. Production
approval requires the acceptance gates recorded with each deployment.

## Privacy and repository boundaries

The repository intentionally contains source code, tests, documentation,
PowerShell launchers, and safe example configuration only. The following are
local artifacts and are excluded by `.gitignore`:

- `.env`, camera credentials, and site-specific camera configuration;
- raw video, extracted frames, evidence, reports, and databases;
- training datasets, intermediate checkpoints, downloaded weights, and virtual environments.

The checksum-verified V2 handover inference bundle is the only committed model bundle.
Its `.pth` file is stored with Git LFS; raw captures and training checkpoints
remain local-only.

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
git lfs install
git lfs pull
Copy-Item .env.example .env
.\setup.ps1
```

The setup script installs the selected PyTorch build, project dependencies, and
the editable `scoop-ai` package. CUDA 13.0 is the default; use `-Compute cu128`
or `-Compute cpu` when appropriate.

After `git lfs pull`, the deployable detector must exist at:

```text
models/ice-cream-item-rfdetr-nano-v2/
|-- checkpoint_best_total.pth
`-- model-manifest.json
```

The application verifies the checkpoint SHA-256 from the manifest before
loading it. Datasets and annotated replay videos are not required for runtime.

## Production camera credentials

Do not place RTSP credentials in `.env`, TOML, command-line arguments, shell
history, screenshots, or source control. Provision the complete source URL in
Windows Credential Manager using the same hierarchical key referenced by the
example camera configuration:

```powershell
.\.venv\Scripts\scoop-ai.exe credential-set `
  --key scoop-ai/shop-01-counter-01/rtsp-url `
  --database D:\ip-camera-ai-data\database\events.sqlite3
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

## Local operator dashboard

Run the loopback-only browser dashboard on the shop computer:

```powershell
.\dashboard.ps1
```

It opens `http://127.0.0.1:8090` and provides daily candidate totals, service
status, evidence review, accept/reject decisions, and item-quantity correction.
The dashboard reads the same local SQLite database and evidence directory as
the Windows edge service; camera video and event data are not uploaded.
`setup.ps1` also creates a **Scoop AI Dashboard** desktop shortcut by default;
use `-SkipDashboardShortcut` on developer machines that do not need it.

On first launch, the client opens a guided setup wizard for shop details,
credential-backed camera connection, live preview, and visual pickup/customer
zone calibration. After setup, monitoring can be started or stopped from the
dashboard. Client data remains under `D:\ip-camera-ai-data` by default.

## Windows client installer

Release builds use PyInstaller plus Inno Setup to create a self-contained
Windows installer:

```powershell
.\build-installer.ps1
```

The installer output is written under `dist\installer`. It creates a desktop
shortcut and a background startup entry. Tagged releases and manual runs of
the `Windows installer` GitHub Actions workflow build the same artifact.

Normal builds are development/pilot artifacts and also write
`dist\installer\release-manifest.json` with the exact Git commit, hashes,
sizes, dirty-tree state, and Authenticode status. A customer release uses the
stricter gate:

```powershell
.\build-installer.ps1 -Release `
  -CertificateThumbprint <authenticode-certificate-thumbprint> `
  -CommercialLicenseAcknowledged
```

The release gate refuses a dirty or untagged tree, an unsigned build, and an
installer above the configured two-billion-byte delivery ceiling. Keep pilot
builds labelled as human-reviewed silent-pilot software.

The Windows runtime currently keeps `setuptools>=78.1.1,<82` because Torch
2.11 requires that upper bound. CI temporarily ignores only
`PYSEC-2026-3447`, a macOS source-distribution Unicode-normalization advisory;
the Windows frozen build neither accepts nor packages untrusted source names.
Remove this exception when the approved Torch runtime permits setuptools 83+
and continue treating every other `pip-audit` finding as blocking.

## Documentation map

- [`docs/PRODUCTION.md`](docs/PRODUCTION.md): secure setup and service operation;
- [`docs/RUNBOOKS.md`](docs/RUNBOOKS.md): operational incident procedures.
