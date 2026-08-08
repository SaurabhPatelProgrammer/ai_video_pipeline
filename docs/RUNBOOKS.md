# Scoop AI incident runbooks

These procedures assume the service settings are under
`D:\ip-camera-ai-data\service` and the operator is using an elevated
PowerShell where service actions are required. Preserve the SQLite database
and logs before any recovery action. Never delete raw evidence to make a health
alarm disappear.

## Camera loss or invalid credentials

1. Check `http://127.0.0.1:8080/health` and confirm the failing component is
   `capture`.
2. Stop the service: `Stop-Service ScoopAIEdge`.
3. Confirm network reachability and that RTSP/ONVIF is enabled on the camera.
4. Re-provision the secret with `scoop-ai credential-set --key <credential-key> --database <events.sqlite3>`;
   do not print the URL in the terminal.
5. Run `scoop-ai camera-check --camera-config <camera.toml> --frames 30`.
6. Start the service and confirm frame age and reconnect counters recover.

## Windows service watchdog and crash recovery

The service watchdog checks capture frame progress and inference completion.
After a bounded stall it marks the component unhealthy and requests a clean
service stop. The Windows SCM failure actions configured by
`install-service.ps1` then restart the process after 60, 120, and 300 seconds;
the failure counter resets after 24 hours to prevent rapid restart thrashing.

Before starting, the service validates the effective account's write access to
the database/evidence directories, configured minimum free disk space, camera
configuration, and the OS credential reference. Invalid credentials or an
unwritable artifact root fail startup immediately.

Run the service under a dedicated Windows service account with only the
following access:

- read/execute access to the application and model bundle;
- read access to the camera network path as required by the site firewall;
- modify access to `D:\ip-camera-ai-data\database`, `evidence`, `logs`, and
  `service`;
- no local administrator rights and no interactive logon.

After installation, verify the SCM policy with:

```powershell
sc.exe qfailure ScoopAIEdge
Get-Service ScoopAIEdge
Invoke-WebRequest http://127.0.0.1:8080/health
```

Do not use `sc.exe failure` to mask a configuration or credential error; fix
the startup validation failure first.

## Health and alert actions

The loopback `/health` endpoint reports readiness/liveness and `/metrics`
returns bounded JSON counters, gauges, and latency summaries. Alert events are
also persisted in `health_events` with a unique `alert_id`; repeated polls do
not create duplicate rows until the condition changes or recovers.

- **Capture degraded / stale heartbeat:** check camera power, network path,
  frame age, and the approved camera view. Run `camera-check`; do not promote
  events from the affected interval until reviewed.
- **Camera reconnect warning (3 failures):** inspect RTSP reachability and
  credentials, then run `camera-check --frames 30`. Keep the service running
  while the built-in reconnect backoff operates.
- **Camera reconnect critical (10 failures):** stop downstream use, verify the
  camera and firewall, and restart only after `camera-check` succeeds.
- **Inference degraded/unhealthy:** inspect GPU memory, model manifest and
  watchdog logs. Re-run the recorded canary before changing the model.
- **Low disk warning (<10 GiB):** preserve logs/database, check backup and
  retention status, and archive approved raw data. Do not manually delete the
  active database or registered evidence.
- **Low disk critical (<2 GiB):** stop the service, protect the database/WAL,
  complete a verified backup or restore drill, then reclaim space according to
  the disk-pressure runbook.
- **Storage/database lock alert:** check concurrent review/backup activity and
  disk latency. Run `database-check` after the condition clears.

## Camera moved, obstructed, or blurred

1. Treat a sustained quality-gate degradation as data invalidity, not a model
   failure. The service intentionally suppresses inference.
2. Restore the approved camera mount, focus, zoom, lighting, and tub visibility.
3. Compare the live view to the deployment reference image.
4. If the physical view intentionally changed, create a new camera calibration
   and validation dataset version; do not loosen thresholds on the live device.

## Calibration drift and zone obstruction

The production camera profile contains an immutable grayscale/edge fingerprint
and the approved tub/serving polygons. The service compares incoming frames to
that fingerprint and suppresses model inference after a sustained drift or
zone obstruction. The dynamic quality reference is never allowed to overwrite
this approved fingerprint.

1. Confirm `/health` shows `capture` degraded and inspect `/metrics` for
   `scene_drift_score`, `tub_zone_changed_fraction`, and
   `serving_zone_changed_fraction`.
2. Inspect the camera mount, focus, zoom, lighting, lens and physical zone
   visibility. Do not lower drift or obstruction thresholds as a first action.
3. If the camera is physically unchanged, clear the obstruction and confirm
   the metrics recover before accepting any affected events.
4. If the view intentionally changed, stop downstream use and run
   `scoop-ai recalibrate` to create a new versioned profile. Update
   `camera.calibration_profile`, validate a recorded canary, and restart.
5. Treat all events from the drift interval as needing manual review; recovery
   requires the explicit recalibration workflow.

## GPU out of memory

1. Stop the service and record the model manifest SHA and recent logs.
2. Close unrelated GPU workloads and restart once.
3. If the fault repeats, restore the last approved smaller-model manifest and
   restart. Do not silently reduce input resolution on an approved manifest.
4. Re-run the recorded benchmark and event acceptance gate before promoting a
   different architecture.

## Disk pressure

1. Stop the service if free space threatens SQLite or evidence writes.
2. Verify retention state in the database before removing any registered file.
3. Archive approved datasets/raw captures to an access-controlled volume.
4. Let the service retention job delete only expired evidence. Never manually
   remove the active database, WAL, or SHM files.
5. Restart and confirm the storage component and database integrity check pass.

## Evidence/database consistency drift

The service runs an automatic reconciliation pass at every startup, before it
opens the camera source. It compares the evidence directory against the
`evidence_artifacts` and `events` tables and only ever repairs bookkeeping —
it never deletes a file that an event still references and never fabricates
event data.

- **Orphan file referenced by an event** (the process crashed after writing
  the evidence file and inserting the event row, but before the evidence row
  committed): the file is re-hashed and registered as `integrity_status:
  valid`. No operator action required; confirm in the logs that the recovered
  path matches the expected session.
- **Orphan file with no matching event**: deleted immediately as leftover
  temp/partial output. Confirm in the logs this was expected (e.g. after a
  crash during frame capture) before assuming it is safe on a device you did
  not expect this on.
- **Database record missing its file** (file lost outside the service, not
  through the retention job): `integrity_status` is set to `missing`, the
  linked event is moved to `needs_review`, and a `degraded` row is written to
  `health_events`. Investigate why the file disappeared (disk corruption,
  manual deletion, antivirus quarantine) before clearing the review flag.
  Evidence removed by the retention job itself is never flagged this way.

Run the same check on demand, independent of a service restart:

```powershell
.\.venv\Scripts\scoop-ai.exe consistency-check `
  --database D:\ip-camera-ai-data\database\events.sqlite3 `
  --evidence-root D:\ip-camera-ai-data\evidence
```

The command prints a JSON report (`orphans`, `missing`, `corrupt`) and exits
non-zero if any list is non-empty. `corrupt` means the file exists but its
size or SHA-256 no longer matches the registered evidence row — treat this as
tampering or disk corruption, preserve the file for analysis, and do not
delete it manually; escalate per [Escalation and return to
service](#escalation-and-return-to-service).

## SQLite recovery

1. Stop the service; never copy a live database without its WAL/SHM state.
2. Preserve the complete `database` directory as incident evidence.
3. Run `scoop-ai database-check --database <events.sqlite3>`.
4. If integrity fails, restore the latest tested backup to a new path and run
   the check again. Point the service to the recovered artifact root only after
   review/audit rows and model registrations have been verified.

## Backup and restore drill

Create backups while the service is running; the command uses SQLite's online
backup API and includes the active WAL state in the consistent snapshot:

```powershell
.\.venv\Scripts\scoop-ai.exe database-backup `
  --database D:\ip-camera-ai-data\database\events.sqlite3 `
  --output-dir D:\ip-camera-ai-data\backups `
  --retain-generations 5
```

Each `.sqlite3` backup has a matching `.metadata.json` file containing the
source database name, schema version, UTC backup time, byte size, SHA-256, and
record counts for events, reviews, ground truth, and registered models. The
cleanup option removes only older recognized backup pairs.

Restore only into a new directory. The restore command never overwrites an
existing `events.sqlite3`, verifies the metadata checksum and schema version,
then runs SQLite integrity and record-count checks:

```powershell
.\.venv\Scripts\scoop-ai.exe database-restore `
  --backup D:\ip-camera-ai-data\backups\events-<generation>.sqlite3 `
  --output-dir D:\ip-camera-ai-data\restore-drill

.\.venv\Scripts\scoop-ai.exe database-check `
  --database D:\ip-camera-ai-data\restore-drill\events.sqlite3
```

For a quarterly restore drill, compare the JSON record counts with the live
database, open the local review application against the restored copy, and
preserve the backup plus restore output until the operator signs off. Never
point the active service at a restore path until the drill and review are
complete.

## Clock drift

1. Recorded sources use media time and should not be corrected in event data.
2. For live cameras, verify Windows Time service health and compare UTC with an
   approved NTP source.
3. Correct the host clock, restart the service, and mark the affected session
   range for manual review. Never rewrite stored event timestamps in place.

## Model rollback

1. Stop the service and retain the failed manifest/checkpoint and logs.
2. Update the Windows service settings JSON to the previous approved manifest.
3. Run `scoop-ai model-manifest` only when creating a new bundle; never edit a
   manifest checksum by hand.
4. Start the service and verify `/health`, `/metrics`, a recorded canary, and
   the database model-version row.

## Escalation and return to service

Keep the system in silent-pilot mode when root cause is unknown, evidence
integrity is uncertain, or an acceptance gate regresses. Return to automated
downstream use only after the incident is documented, the deterministic replay
passes, and a reviewer signs off the affected period.
## Phase 6: deterministic replay and fault drills

Replay a recorded session through the same timestamp-driven detector, tracker,
quality gate, and deposit state machine used by the service:

```powershell
scoop-ai replay `
  --video D:\captures\shop-01-session.mp4 `
  --service-config configs\service.example.toml `
  --camera-config configs\cameras\shop-01-counter-01.example.toml `
  --checkpoint-manifest D:\ip-camera-ai-data\models\approved\manifest.json `
  --output D:\captures\reports\shop-01-replay.json
```

The report records input, manifest, and TOML hashes, event timestamp hashes,
the canonical `events_sha256`/`replay_checksum`, and observed wall/CPU/memory
benchmarks. Re-running the same inputs must produce identical event and replay
checksums; timing fields are diagnostic and are not part of the checksum.

The deterministic fault primitives in `scoop_ai.testing` support controlled
GPU OOM, SQLite lock, disk-full, frame-dropout, and network-latency drills.
Run the complete recovery suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_phase6.py
```

For a production drill, inject one fault at a time, confirm the service
reconnects/resumes, then verify `scoop-ai database-check` and the event count.
An event retry with the same event ID must be an idempotent no-op, never a
duplicate or conflicting payload.

## Phase 7: model approval and deployment governance

Create a manifest with the reviewer identity and detached approval signature:

```powershell
scoop-ai model-manifest `
  --checkpoint D:\models\model-v2\model.pth `
  --output D:\models\model-v2\manifest.json `
  --architecture nano `
  --dataset-version dataset-2026-08 `
  --model-version scoop-rfdetr-nano-v2 `
  --approved-by platform-reviewer `
  --reviewer-signature sha256:<signature>
```

Register and promote only after the manifest verifies against the approved
reviewer list:

```powershell
scoop-ai model-register --manifest D:\models\model-v2\manifest.json `
  --database D:\ip-camera-ai-data\database\events.sqlite3 `
  --approved-reviewer platform-reviewer
scoop-ai model-promote --manifest D:\models\model-v2\manifest.json `
  --database D:\ip-camera-ai-data\database\events.sqlite3 `
  --camera-id shop-01-counter-01 --approved-reviewer platform-reviewer
scoop-ai model-rollback --database D:\ip-camera-ai-data\database\events.sqlite3 `
  --camera-id shop-01-counter-01 --approved-reviewer platform-reviewer
```

Production startup rejects missing/invalid approval, missing active model
registration, manifest/active-version mismatch, missing zones, and camera
dimensions smaller than the model input resolution. The selected model version
is persisted in the session metadata and exposed in the inference health details.

## Phase 8: privacy, audit, and access controls

Administrative actions are recorded in the immutable `audit_logs` table. This
covers backups/restores, credential changes, retention-rule changes, and manual
evidence deletion. Use `scoop-ai evidence-delete` only with the explicit
`--confirm DELETE` flag; normal retention deletion is separately marked as a
service audit event.

For privacy-safe sharing, export a blurred image using tracked normalized boxes
or polygon zones:

```powershell
scoop-ai evidence-export --input frame.jpg --output frame-redacted.jpg `
  --box 0.10,0.15,0.40,0.90
```

Before installing or changing the Windows service, validate explicit service
account ACLs on writable directories:

```powershell
.\scripts\validate-acls.ps1 `
  -ArtifactRoot D:\ip-camera-ai-data `
  -ServiceAccount 'NT SERVICE\ScoopAIEdge'
```

Logs redact RTSP URL credentials, secret-like fields, and exception tracebacks
before serialization. Never paste raw camera URLs, database files, or
unredacted evidence into tickets or chat.

## Phase 9: controlled silent pilot

The service configuration must explicitly keep billing and automatic exports
disabled. Generate the daily pilot report from SQLite:

```powershell
scoop-ai generate-pilot-report `
  --database D:\ip-camera-ai-data\database\events.sqlite3 `
  --camera-id shop-01-counter-01 `
  --output D:\ip-camera-ai-data\reports\pilot-2026-08-08.json
```

The report includes session uptime, persisted frame-rate/quality telemetry,
candidate event count, manual review count, reviewer agreement rate, and
degraded/unhealthy alert counts. It is review-only telemetry; no downstream
billing or automatic export adapter is connected in the pilot.

Rehearse backup, restore, integrity validation, and optional rollback on a
restored copy before pilot launch:

```powershell
.\scripts\rehearse-pilot.ps1 `
  -Database D:\ip-camera-ai-data\database\events.sqlite3 `
  -BackupDirectory D:\ip-camera-ai-data\rehearsal\backups `
  -RestoreDirectory D:\ip-camera-ai-data\rehearsal\restored `
  -CameraId shop-01-counter-01 `
  -ApprovedReviewer platform-reviewer
```

Keep the restored rehearsal database isolated from the running service. Record
the generated report and rehearsal output as the pilot evidence package.

## Phase 10: controlled downstream export

Downstream export is disabled by default. Enable it only in a separately
approved service configuration with an HTTPS endpoint and an OS credential
reference for the HMAC signing key. Only events with an accepted manual review
enter `event_outbox`; raw AI candidates never become export transactions.

The worker sends an event UUID as `Idempotency-Key`, signs the canonical export
payload, retries failures with exponential backoff, and moves exhausted events
to `dead_letter`. Re-queue a reviewed dead-letter event after the endpoint is
fixed:

```powershell
scoop-ai outbox-retry `
  --database D:\ip-camera-ai-data\database\events.sqlite3 `
  --event-id <event-uuid>
```

Inspect the outbox states before approving any downstream reconciliation. The
accepted-review record, outbox row, export signature, and downstream response
must all be retained as separate audit evidence. Never enable export on the
silent-pilot configuration.

## Phase 11: final production-gate readiness

Run the readiness audit only after collecting production evidence:

```powershell
scoop-ai production-readiness-check `
  --service-config configs\service.example.toml `
  --camera-config configs\cameras\shop-01-counter-01.example.toml `
  --database D:\ip-camera-ai-data\database\events.sqlite3 `
  --dataset D:\ip-camera-ai-data\datasets\production.manifest.json `
  --backup D:\ip-camera-ai-data\rehearsal\backups\verified.sqlite3 `
  --output D:\ip-camera-ai-data\reports\production-gate.json
```

The command returns exit code `1` whenever a blocking gate is missing. It
checks billing/export safety, calibrated zones, database integrity, active
approved model, outbox dead letters, representative dataset evidence, backup
drill evidence, and separate downstream approval. Until every check passes,
keep the product label as `production-oriented review and silent-pilot edge
analytics`, not billing-grade autonomous counting.
