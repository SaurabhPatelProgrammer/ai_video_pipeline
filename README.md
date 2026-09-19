# IP Camera AI MVP

Fixed-camera video analytics for detecting completed ice-cream serving events.
The repository contains a production-oriented Windows edge-service foundation,
diagnostic OpenCV tools, dataset validation, model-manifest verification,
review workflows, and experimental motion baselines.

The current motion profiles are review-only prototypes. They are not billing-
grade counters and must not drive autonomous commercial decisions. The canonical
production path is the installable `scoop_ai` package.

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
- an NVIDIA driver and GPU, or a 4-core-or-better CPU (see [Compute targets](#compute-targets));
- a reachable webcam, RTSP camera, HTTP stream, or recorded video file.

## Compute targets

The client runs on GPU and CPU machines from the same bundle. `camera.device`
selects the target: `auto` (default) uses the GPU when a working NVIDIA runtime
is present and falls back to the CPU otherwise, and `cuda` or `cpu` pin it.
A pinned `cuda` on a machine without a GPU logs a warning and still runs on the
CPU rather than refusing to start.

Report what a machine will actually use:

```powershell
.\.venv\Scripts\scoop-ai.exe compute-check
.\.venv\Scripts\scoop-ai.exe compute-check --device cpu --json
```

Measured RF-DETR nano latency at 1280x720 (RTX 3050 laptop GPU, 12-thread CPU):

| Target | First frame | Steady state | Sustainable rate |
| ------ | ----------- | ------------ | ---------------- |
| GPU    | 12.5 s      | 96 ms        | ~10 fps          |
| CPU    | 6.5 s       | 212 ms       | ~4.7 fps         |

Because the analysis rate must leave headroom for capture, the quality gate, and
evidence writes — on shop hardware weaker than a development machine — the
service caps analysis at 10 fps on GPU and 2 fps on CPU, and logs when it lowers
a configured rate. CPU inference also reserves one core so capture and the user
interface stay responsive. A handover takes well over a second, so 2 fps still
samples each one several times.

Do not raise `camera.analysis_fps` above these ceilings on a CPU machine; frames
would queue faster than they drain and evidence timestamps would drift behind the
camera clock.

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
the editable `scoop-ai` package. It probes for an NVIDIA driver and installs the
CUDA 13.0 runtime when one is present, or the CPU runtime when it is not. Use
`-Compute cu130`, `-Compute cu128`, or `-Compute cpu` to override the probe.
For a development or validation machine, add `-IncludeDevTools` to install the
test and lint dependencies as well.

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
view. For the handover pipeline, `tub` is the pickup/preparation zone and
`serving` is the customer handover zone.

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
configured for pytest in CI. Run `.\setup.ps1 -IncludeDevTools` first when the
virtual environment was created with the normal production setup:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src tests scripts
```

## Desktop application

The client is a desktop application. Launching it opens a native window — no
browser, address bar, or URL — with the operator dashboard rendered inside it:

```powershell
.\desktop.ps1
.\desktop.ps1 -SoftwareRendering   # Remote Desktop or a machine without a display GPU
```

`setup.ps1` creates a **Scoop AI** desktop shortcut and a startup entry that
launches minimised to the notification area. Use `-SkipDashboardShortcut` on
developer machines that do not need them.

The window provides three pages, reached from the sidebar:

- **Today** — daily candidate totals, service and compute status, and evidence
  review with accept/reject decisions and item-quantity correction;
- **Camera** — a live view of the configured camera with the calibrated pickup
  and customer zones drawn over it, the saved configuration, and a deliberate
  route back into the setup wizard;
- **System** — inference device, PyTorch build, analysis-rate ceiling, and the
  local data, database, and evidence paths.

Until a camera is configured, every route leads to the setup wizard: shop
details, credential-backed camera connection, a still preview, and visual
pickup/customer zone calibration.

Network discovery finds ONVIF cameras and then asks the chosen camera, over
ONVIF, for its own RTSP address. The operator supplies only the camera username
and password; the vendor's stream path is never guessed, because a guessed path
produces a URL that cannot connect. The password is used inside the service to
authenticate and open the stream, and is written to Windows Credential Manager
— it is never returned to the page, logged, or stored in configuration. Manual
RTSP entry remains available for cameras without ONVIF, and both fields can be
revealed to check what was typed. After that the wizard is only reached
explicitly, through **Re-run camera setup** on the Camera page — a configured
camera is never sent back through onboarding.

The notification-area icon starts and stops monitoring without opening the
window.

### Live view and camera load

The Camera page holds one capture open and serves whichever frame arrived last,
so a viewer polling several times a second costs almost nothing. Opening the
stream takes a few seconds on a network camera; every frame after that is served
from memory. The capture is released a few seconds after the last request, so
closing or pausing the page hands the camera back.

The live view is a second connection to the camera. Most cameras accept several,
but a camera limited to one stream will refuse the preview while monitoring is
running, and the page says so rather than failing silently. Pause the live view
or stop monitoring in that case.

Closing the window hides it to the notification area so monitoring keeps
running; **Quit** from the tray menu stops monitoring and exits. Launching the
shortcut a second time raises the running window instead of starting a second
copy. The embedded view refuses to navigate anywhere except the loopback client,
so evidence cannot be steered off-box by a page defect.

The dashboard reads the same local SQLite database and evidence directory as the
Windows edge service; camera video and event data are not uploaded. Client data
remains under `D:\ip-camera-ai-data` by default.

### Browser dashboard

The loopback browser dashboard remains available for support and for machines
that cannot host the embedded window:

```powershell
.\dashboard.ps1
```

It serves `http://127.0.0.1:8090` with the same interface. The desktop launcher
falls back to it automatically if QtWebEngine is unavailable.

## Windows client installer

Release builds use PyInstaller plus Inno Setup to create a self-contained
Windows installer:

```powershell
.\build-installer.ps1
```

The installer output is written under `dist\installer`. It creates a desktop
shortcut that opens the desktop window and a startup entry that launches it
minimised to the notification area. Tagged releases and manual runs of
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
