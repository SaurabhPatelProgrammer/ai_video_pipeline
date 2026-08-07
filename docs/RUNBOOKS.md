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
4. Re-provision the secret with `scoop-ai credential-set --key <credential-key>`;
   do not print the URL in the terminal.
5. Run `scoop-ai camera-check --camera-config <camera.toml> --frames 30`.
6. Start the service and confirm frame age and reconnect counters recover.

## Camera moved, obstructed, or blurred

1. Treat a sustained quality-gate degradation as data invalidity, not a model
   failure. The service intentionally suppresses inference.
2. Restore the approved camera mount, focus, zoom, lighting, and tub visibility.
3. Compare the live view to the deployment reference image.
4. If the physical view intentionally changed, create a new camera calibration
   and validation dataset version; do not loosen thresholds on the live device.

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

## SQLite recovery

1. Stop the service; never copy a live database without its WAL/SHM state.
2. Preserve the complete `database` directory as incident evidence.
3. Run `scoop-ai database-check --database <events.sqlite3>`.
4. If integrity fails, restore the latest tested backup to a new path and run
   the check again. Point the service to the recovered artifact root only after
   review/audit rows and model registrations have been verified.

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
