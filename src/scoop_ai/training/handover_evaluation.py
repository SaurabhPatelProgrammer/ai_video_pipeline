"""Score handover replay reports against human-reviewed ground truth.

A replay report says what the system counted. Only a person watching the
annotated video knows what actually happened. This module joins the two so the
pilot has one auditable number instead of an impression, and keeps two
different questions apart:

* **transactions** - how many handovers happened, matched on time;
* **item quantity** - how many ice creams changed hands, which can exceed the
  transaction count when several items are handed over together.

Nothing here re-runs inference. It reads finished reports, so re-scoring after
a ground-truth correction is instant and never changes the predictions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path


TRUTH_SCHEMA_VERSION = 1


class HandoverEvaluationError(ValueError):
    """Raised when a report or ground-truth file cannot be trusted."""


@dataclass(frozen=True, slots=True)
class TruthTransaction:
    timestamp: float
    quantity: int = 1
    note: str = ""

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise HandoverEvaluationError("truth timestamp must be finite and non-negative")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise HandoverEvaluationError("truth quantity must be an integer")
        if self.quantity < 1:
            raise HandoverEvaluationError("truth quantity must be at least 1")


@dataclass(frozen=True, slots=True)
class PredictedEvent:
    timestamp: float
    event_id: int
    route: str
    confidence: float
    evidence_file: str


@dataclass(frozen=True, slots=True)
class ReplaySession:
    session: str
    report_path: Path
    video: str
    model_version: str
    events: tuple[PredictedEvent, ...]


@dataclass(frozen=True, slots=True)
class MatchedPair:
    truth_timestamp: float
    predicted_timestamp: float
    time_error_seconds: float
    event_id: int
    route: str
    truth_quantity: int


@dataclass
class SessionScore:
    session: str
    video: str
    truth_transactions: int
    detected_events: int
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    truth_quantity: int = 0
    quantity_gap: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_time_error_seconds: float | None = None
    max_time_error_seconds: float | None = None
    matches: list[MatchedPair] = field(default_factory=list)
    missed_truth_timestamps: list[float] = field(default_factory=list)
    extra_event_timestamps: list[float] = field(default_factory=list)
    review_actions: list[str] = field(default_factory=list)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def load_replay_session(report_path: str | Path) -> ReplaySession:
    """Read one ``report.json`` written by ``run_handover_replay``."""

    path = Path(report_path).resolve()
    if not path.is_file():
        raise HandoverEvaluationError(f"replay report not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoverEvaluationError(f"could not read replay report {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        raise HandoverEvaluationError(f"{path} is not a handover replay report")

    events: list[PredictedEvent] = []
    for index, item in enumerate(data["events"]):
        if not isinstance(item, dict) or "timestamp" not in item:
            raise HandoverEvaluationError(f"{path}: event {index} has no timestamp")
        events.append(
            PredictedEvent(
                timestamp=float(item["timestamp"]),
                event_id=int(item.get("event_id", index + 1)),
                route=str(item.get("route", "unknown")),
                confidence=float(item.get("confidence", 0.0)),
                evidence_file=str(item.get("evidence_file", "")),
            )
        )
    events.sort(key=lambda event: (event.timestamp, event.event_id))
    return ReplaySession(
        # The replay output directory names the session in every existing
        # artifact layout, which keeps truth files readable by a reviewer.
        session=path.parent.name,
        report_path=path,
        video=str(data.get("video", "")),
        model_version=str(data.get("model_version", "")),
        events=tuple(events),
    )


def build_truth_template(report_paths: list[str | Path]) -> dict[str, object]:
    """Pre-fill a ground-truth file with the detections a reviewer must judge.

    Every detected event becomes a candidate row. The reviewer deletes rows that
    were not real handovers, corrects ``quantity`` where several items went out
    together, and appends rows for handovers the system missed entirely.
    """

    if not report_paths:
        raise HandoverEvaluationError("at least one replay report is required")
    sessions: dict[str, object] = {}
    for report_path in report_paths:
        session = load_replay_session(report_path)
        if session.session in sessions:
            raise HandoverEvaluationError(f"duplicate session name: {session.session}")
        sessions[session.session] = {
            "video": session.video,
            "transactions": [
                {
                    "timestamp": round(event.timestamp, 3),
                    "quantity": 1,
                    "note": f"candidate from event {event.event_id} ({event.route}); confirm or delete",
                }
                for event in session.events
            ],
        }
    return {
        "schema_version": TRUTH_SCHEMA_VERSION,
        "instructions": (
            "Watch annotated.mp4 for each session. Delete rows that were not real "
            "handovers, fix quantity when several items were handed over at once, "
            "and add a row for every handover the system missed."
        ),
        "sessions": sessions,
    }


def load_truth(truth_path: str | Path) -> dict[str, tuple[str, tuple[TruthTransaction, ...]]]:
    path = Path(truth_path).resolve()
    if not path.is_file():
        raise HandoverEvaluationError(f"ground-truth file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoverEvaluationError(f"could not read ground truth {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("sessions"), dict):
        raise HandoverEvaluationError("ground truth must contain a 'sessions' object")
    if int(data.get("schema_version", TRUTH_SCHEMA_VERSION)) != TRUTH_SCHEMA_VERSION:
        raise HandoverEvaluationError(
            f"unsupported ground-truth schema_version: {data.get('schema_version')}"
        )

    output: dict[str, tuple[str, tuple[TruthTransaction, ...]]] = {}
    for session, details in data["sessions"].items():
        if not isinstance(details, dict) or not isinstance(details.get("transactions"), list):
            raise HandoverEvaluationError(f"{session}: 'transactions' must be a list")
        transactions = []
        for index, row in enumerate(details["transactions"]):
            if not isinstance(row, dict) or "timestamp" not in row:
                raise HandoverEvaluationError(f"{session}: transaction {index} has no timestamp")
            transactions.append(
                TruthTransaction(
                    timestamp=float(row["timestamp"]),
                    quantity=int(row.get("quantity", 1)),
                    note=str(row.get("note", "")),
                )
            )
        transactions.sort(key=lambda item: item.timestamp)
        output[str(session)] = (str(details.get("video", "")), tuple(transactions))
    return output


def _match(
    truths: tuple[TruthTransaction, ...],
    predictions: tuple[PredictedEvent, ...],
    tolerance_seconds: float,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one nearest matching; closest pairs win first."""

    candidates: list[tuple[float, int, int]] = []
    for truth_index, truth in enumerate(truths):
        for prediction_index, prediction in enumerate(predictions):
            error = abs(prediction.timestamp - truth.timestamp)
            if error <= tolerance_seconds:
                candidates.append((error, truth_index, prediction_index))
    candidates.sort()
    used_truths: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for error, truth_index, prediction_index in candidates:
        if truth_index in used_truths or prediction_index in used_predictions:
            continue
        used_truths.add(truth_index)
        used_predictions.add(prediction_index)
        matches.append((truth_index, prediction_index, error))
    matches.sort(key=lambda item: item[0])
    return matches


def _score_session(
    session: ReplaySession,
    truths: tuple[TruthTransaction, ...],
    tolerance_seconds: float,
) -> SessionScore:
    matches = _match(truths, session.events, tolerance_seconds)
    matched_truths = {truth_index for truth_index, _, _ in matches}
    matched_predictions = {prediction_index for _, prediction_index, _ in matches}
    errors = [error for _, _, error in matches]

    true_positives = len(matches)
    false_positives = len(session.events) - true_positives
    false_negatives = len(truths) - true_positives
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    truth_quantity = sum(truth.quantity for truth in truths)

    score = SessionScore(
        session=session.session,
        video=session.video,
        truth_transactions=len(truths),
        detected_events=len(session.events),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truth_quantity=truth_quantity,
        # One event currently means one counted item, so the gap exposes the
        # known simultaneous-handover undercount.
        quantity_gap=len(session.events) - truth_quantity,
        precision=precision,
        recall=recall,
        f1=round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0,
        mean_time_error_seconds=round(sum(errors) / len(errors), 3) if errors else None,
        max_time_error_seconds=round(max(errors), 3) if errors else None,
        matches=[
            MatchedPair(
                truth_timestamp=truths[truth_index].timestamp,
                predicted_timestamp=session.events[prediction_index].timestamp,
                time_error_seconds=round(error, 3),
                event_id=session.events[prediction_index].event_id,
                route=session.events[prediction_index].route,
                truth_quantity=truths[truth_index].quantity,
            )
            for truth_index, prediction_index, error in matches
        ],
        missed_truth_timestamps=[
            truth.timestamp
            for index, truth in enumerate(truths)
            if index not in matched_truths
        ],
        extra_event_timestamps=[
            event.timestamp
            for index, event in enumerate(session.events)
            if index not in matched_predictions
        ],
    )
    score.review_actions = [
        f"missed handover near {value:.3f}s - re-watch and consider adding training frames"
        for value in score.missed_truth_timestamps
    ] + [
        f"extra count near {value:.3f}s - re-watch and consider adding a negative frame"
        for value in score.extra_event_timestamps
    ] + [
        f"quantity undercount at {match.truth_timestamp:.3f}s "
        f"({match.truth_quantity} items counted as 1)"
        for match in score.matches
        if match.truth_quantity > 1
    ]
    return score


def evaluate_handovers(
    report_paths: list[str | Path],
    truth_path: str | Path,
    *,
    tolerance_seconds: float = 7.0,
) -> dict[str, object]:
    """Score every replay report against its reviewed ground-truth session."""

    if not report_paths:
        raise HandoverEvaluationError("at least one replay report is required")
    if not isfinite(tolerance_seconds) or tolerance_seconds <= 0:
        raise HandoverEvaluationError("tolerance_seconds must be finite and positive")

    truth = load_truth(truth_path)
    sessions = [load_replay_session(path) for path in report_paths]
    missing = sorted({session.session for session in sessions} - set(truth))
    if missing:
        raise HandoverEvaluationError(
            f"ground truth has no sessions named {missing}; available: {sorted(truth)}"
        )

    scores = [
        _score_session(session, truth[session.session][1], tolerance_seconds)
        for session in sorted(sessions, key=lambda item: item.session)
    ]
    true_positives = sum(score.true_positives for score in scores)
    false_positives = sum(score.false_positives for score in scores)
    false_negatives = sum(score.false_negatives for score in scores)
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    all_errors = [match.time_error_seconds for score in scores for match in score.matches]
    model_versions = sorted({session.model_version for session in sessions if session.model_version})

    return {
        "schema_version": 1,
        "tolerance_seconds": tolerance_seconds,
        "model_versions": model_versions,
        "overall": {
            "sessions": len(scores),
            "truth_transactions": sum(score.truth_transactions for score in scores),
            "detected_events": sum(score.detected_events for score in scores),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0,
            "truth_quantity": sum(score.truth_quantity for score in scores),
            "quantity_gap": sum(score.quantity_gap for score in scores),
            "mean_time_error_seconds": round(sum(all_errors) / len(all_errors), 3) if all_errors else None,
            "max_time_error_seconds": round(max(all_errors), 3) if all_errors else None,
        },
        "sessions": [asdict(score) for score in scores],
    }


def format_summary(result: dict[str, object]) -> str:
    """Render a short plain-text summary for an operator's terminal."""

    overall = result["overall"]
    assert isinstance(overall, dict)
    lines = [
        "Handover evaluation",
        f"  tolerance          : {result['tolerance_seconds']}s",
        f"  sessions           : {overall['sessions']}",
        f"  real transactions  : {overall['truth_transactions']}",
        f"  detected events    : {overall['detected_events']}",
        f"  correct / extra / missed : "
        f"{overall['true_positives']} / {overall['false_positives']} / {overall['false_negatives']}",
        f"  precision / recall / f1  : "
        f"{overall['precision']:.3f} / {overall['recall']:.3f} / {overall['f1']:.3f}",
        f"  real items         : {overall['truth_quantity']} "
        f"(counted-minus-real gap: {overall['quantity_gap']:+d})",
    ]
    if overall["mean_time_error_seconds"] is not None:
        lines.append(
            f"  timing error       : mean {overall['mean_time_error_seconds']}s, "
            f"max {overall['max_time_error_seconds']}s"
        )
    sessions = result["sessions"]
    assert isinstance(sessions, list)
    for score in sessions:
        lines.append(
            f"  - {score['session']}: {score['true_positives']} correct, "
            f"{score['false_positives']} extra, {score['false_negatives']} missed"
        )
        for action in score["review_actions"]:
            lines.append(f"      * {action}")
    return "\n".join(lines)
