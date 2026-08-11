# Measuring handover accuracy

A replay report says what the system counted. Only a reviewer watching the
annotated video knows what actually happened. This procedure joins the two and
produces one auditable number per validation round.

Use it for every model change and every new batch of validation footage.

## What the numbers mean

Two different questions are scored separately, because the current detector can
merge two ice creams handed over together into one tracked item:

- **transactions** - how many handovers happened. Matched on time.
- **item quantity** - how many ice creams changed hands. A negative
  `quantity_gap` means the system counted fewer items than were actually served.

`precision` is how much of what the system counted was real. `recall` is how
much of what was real the system found.

## Collecting validation footage

The result is only as honest as the footage. A validation session must be:

- recorded on the **same fixed camera and angle** as production;
- from a **day, time, or shift that contributed no training frames** - reusing
  training footage inflates every number below and makes the round worthless;
- accompanied by **no-handover recordings**, so false positives are measurable.

Aim for 20-50 sessions across different lighting, staff, and traffic levels
before treating any figure as representative.

## Procedure

### 1. Replay every session

Use identical settings for every session in a round, and keep the output
directories named after the session - the evaluator reads the session name from
the directory.

```powershell
.\.venv\Scripts\scoop-ai.exe handover-replay `
  --video "D:\validation\2026-08-14-morning.mp4" `
  --checkpoint-manifest "models\ice-cream-item-rfdetr-nano-v2\model-manifest.json" `
  --zone-profile "configs\handover-calibration.json" `
  --output-dir "artifacts\validation\2026-08-14-morning"
```

### 2. Create the reviewer worksheet

```powershell
.\.venv\Scripts\scoop-ai.exe handover-truth-template `
  --report artifacts\validation\2026-08-14-morning\report.json `
  --report artifacts\validation\2026-08-14-evening\report.json `
  --output artifacts\validation\ground-truth.json
```

Every detection becomes a candidate row. The command refuses to overwrite an
existing worksheet, so reviewed work cannot be lost by re-running it.

### 3. Review the annotated video

Watch `annotated.mp4` for each session start to finish. Do not grade from the
event list alone - a missed handover is invisible there.

For each session, edit the worksheet:

- **delete** rows that were not real handovers;
- **fix `quantity`** where several items were handed over at once;
- **add** a row for every handover the system missed, with your observed
  timestamp;
- keep a short `note` on anything unusual (occlusion, unusual product, staff
  reaching back in).

```json
{
  "schema_version": 1,
  "sessions": {
    "2026-08-14-morning": {
      "video": "2026-08-14-morning.mp4",
      "transactions": [
        {"timestamp": 26.0, "quantity": 1, "note": "normal handover"},
        {"timestamp": 201.0, "quantity": 2, "note": "two cones together"}
      ]
    }
  }
}
```

A session with no handovers keeps an empty `transactions` list. That is the
case that measures false positives, so do not omit it.

### 4. Score

```powershell
.\.venv\Scripts\scoop-ai.exe handover-evaluate `
  --report artifacts\validation\2026-08-14-morning\report.json `
  --report artifacts\validation\2026-08-14-evening\report.json `
  --truth artifacts\validation\ground-truth.json `
  --tolerance 7.0 `
  --output artifacts\validation\score.json
```

Add `--json` for the full machine-readable result, and
`--require-precision` / `--require-recall` to make the command exit non-zero
below a threshold in a scripted gate.

`--tolerance` is how far a detection may sit from the reviewer's timestamp and
still count as the same handover. Events are confirmed on departure, so a few
seconds of lag is normal; 7 seconds is a reasonable starting point. Report the
tolerance alongside any accuracy figure, because the figure is meaningless
without it.

## Reading the output

```text
Handover evaluation
  tolerance          : 7.0s
  sessions           : 3
  real transactions  : 11
  detected events    : 11
  correct / extra / missed : 11 / 0 / 0
  precision / recall / f1  : 1.000 / 1.000 / 1.000
  real items         : 12 (counted-minus-real gap: -1)
  timing error       : mean 0.4s, max 1.2s
  - release-v2-video-1: 3 correct, 0 extra, 0 missed
      * quantity undercount at 201.277s (2 items counted as 1)
```

`review_actions` in the JSON lists each missed handover, extra count, and
quantity undercount with its timestamp, so the next labelling pass can go
straight to those moments in the video.

## Turning results into the next model

- **missed handovers** - annotate frames from those moments; they are the
  hardest positives available.
- **extra counts** - annotate those frames as negatives.
- **quantity undercounts** - annotate each item with its own tight box.

Rebuild the dataset with `--split-mode source-session` so whole videos stay
inside one split, then retrain and re-run this procedure unchanged. Comparing
two models is only valid when footage, zones, settings, and tolerance are
identical.

## Handling of the files

Ground-truth worksheets and score reports embed source video paths and
timestamps, and live under `artifacts/`, which `.gitignore` excludes. Keep them
with the footage they describe and do not commit them.
