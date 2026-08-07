"""Deterministic timestamp/container matching and business event metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from statistics import mean


@dataclass(frozen=True, slots=True)
class EvaluationEvent:
    session_id: str
    timestamp: float
    container_id: str | int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id cannot be empty")
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("event timestamp must be finite and non-negative")
        if isinstance(self.container_id, str) and not self.container_id.strip():
            raise ValueError("container_id cannot be empty")


@dataclass(frozen=True, slots=True)
class EventMatch:
    prediction_index: int
    truth_index: int
    absolute_time_error_seconds: float


@dataclass(frozen=True, slots=True)
class EventMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    wrong_container_assignments: int
    precision: float
    recall: float
    f1: float
    wrong_container_rate: float
    exact_container_count_accuracy: float
    mean_time_error_seconds: float | None
    max_time_error_seconds: float | None
    false_positives_per_hour: float | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: EventMetrics
    matches: tuple[EventMatch, ...]
    wrong_container_matches: tuple[EventMatch, ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_truth_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": asdict(self.metrics),
            "matches": [asdict(item) for item in self.matches],
            "wrong_container_matches": [
                asdict(item) for item in self.wrong_container_matches
            ],
            "unmatched_prediction_indices": list(self.unmatched_prediction_indices),
            "unmatched_truth_indices": list(self.unmatched_truth_indices),
        }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _match_same_container(
    predictions: list[EvaluationEvent],
    truths: list[EvaluationEvent],
    tolerance_seconds: float,
) -> list[EventMatch]:
    prediction_groups: dict[tuple[str, str], list[tuple[int, EvaluationEvent]]] = defaultdict(list)
    truth_groups: dict[tuple[str, str], list[tuple[int, EvaluationEvent]]] = defaultdict(list)
    for index, event in enumerate(predictions):
        prediction_groups[(event.session_id, str(event.container_id))].append((index, event))
    for index, event in enumerate(truths):
        truth_groups[(event.session_id, str(event.container_id))].append((index, event))

    matches: list[EventMatch] = []
    for key in sorted(set(prediction_groups) & set(truth_groups)):
        predicted = sorted(prediction_groups[key], key=lambda item: (item[1].timestamp, item[0]))
        expected = sorted(truth_groups[key], key=lambda item: (item[1].timestamp, item[0]))
        prediction_cursor = 0
        truth_cursor = 0
        while prediction_cursor < len(predicted) and truth_cursor < len(expected):
            prediction_index, prediction = predicted[prediction_cursor]
            truth_index, truth = expected[truth_cursor]
            difference = prediction.timestamp - truth.timestamp
            if abs(difference) <= tolerance_seconds:
                matches.append(
                    EventMatch(prediction_index, truth_index, abs(difference))
                )
                prediction_cursor += 1
                truth_cursor += 1
            elif difference < -tolerance_seconds:
                prediction_cursor += 1
            else:
                truth_cursor += 1
    return sorted(matches, key=lambda item: (item.truth_index, item.prediction_index))


def _match_wrong_containers(
    predictions: list[EvaluationEvent],
    truths: list[EvaluationEvent],
    prediction_indices: set[int],
    truth_indices: set[int],
    tolerance_seconds: float,
) -> list[EventMatch]:
    candidates: list[tuple[float, int, int]] = []
    for prediction_index in prediction_indices:
        prediction = predictions[prediction_index]
        for truth_index in truth_indices:
            truth = truths[truth_index]
            if prediction.session_id != truth.session_id:
                continue
            if str(prediction.container_id) == str(truth.container_id):
                continue
            error = abs(prediction.timestamp - truth.timestamp)
            if error <= tolerance_seconds:
                candidates.append((error, prediction_index, truth_index))
    matches: list[EventMatch] = []
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    for error, prediction_index, truth_index in sorted(candidates):
        if prediction_index in used_predictions or truth_index in used_truths:
            continue
        used_predictions.add(prediction_index)
        used_truths.add(truth_index)
        matches.append(EventMatch(prediction_index, truth_index, error))
    return matches


def evaluate_events(
    predictions: list[EvaluationEvent],
    truths: list[EvaluationEvent],
    *,
    tolerance_seconds: float = 0.75,
    observed_duration_seconds: float | None = None,
) -> EvaluationResult:
    """Evaluate event timing and container assignment without double matching.

    A true positive must be inside the timestamp tolerance *and* assigned to the
    correct session-local container. A temporally matching event assigned to the
    wrong container remains one false positive plus one false negative, and is
    additionally reported as a wrong-container assignment.
    """

    if not isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be finite and non-negative")
    if observed_duration_seconds is not None and (
        not isfinite(observed_duration_seconds) or observed_duration_seconds <= 0
    ):
        raise ValueError("observed_duration_seconds must be finite and positive")

    matches = _match_same_container(predictions, truths, tolerance_seconds)
    matched_predictions = {item.prediction_index for item in matches}
    matched_truths = {item.truth_index for item in matches}
    unmatched_predictions = set(range(len(predictions))) - matched_predictions
    unmatched_truths = set(range(len(truths))) - matched_truths
    wrong_matches = _match_wrong_containers(
        predictions,
        truths,
        unmatched_predictions,
        unmatched_truths,
        tolerance_seconds,
    )

    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(truths) - true_positives
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    wrong_count = len(wrong_matches)
    assignment_total = true_positives + wrong_count
    wrong_container_rate = wrong_count / assignment_total if assignment_total else 0.0

    prediction_counts = Counter(
        (event.session_id, str(event.container_id)) for event in predictions
    )
    truth_counts = Counter((event.session_id, str(event.container_id)) for event in truths)
    count_keys = set(prediction_counts) | set(truth_counts)
    exact_count_accuracy = (
        sum(prediction_counts[key] == truth_counts[key] for key in count_keys)
        / len(count_keys)
        if count_keys
        else 1.0
    )

    errors = [item.absolute_time_error_seconds for item in matches]
    fp_per_hour = (
        false_positives * 3600.0 / observed_duration_seconds
        if observed_duration_seconds is not None
        else None
    )
    metrics = EventMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        wrong_container_assignments=wrong_count,
        precision=precision,
        recall=recall,
        f1=f1,
        wrong_container_rate=wrong_container_rate,
        exact_container_count_accuracy=exact_count_accuracy,
        mean_time_error_seconds=mean(errors) if errors else None,
        max_time_error_seconds=max(errors) if errors else None,
        false_positives_per_hour=fp_per_hour,
    )
    return EvaluationResult(
        metrics=metrics,
        matches=tuple(matches),
        wrong_container_matches=tuple(wrong_matches),
        unmatched_prediction_indices=tuple(sorted(unmatched_predictions)),
        unmatched_truth_indices=tuple(sorted(unmatched_truths)),
    )
