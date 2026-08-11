"""Tests for scoring handover replays against reviewed ground truth."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.training import (  # noqa: E402
    HandoverEvaluationError,
    build_truth_template,
    evaluate_handovers,
    format_summary,
    load_replay_session,
)


def _write_report(directory: Path, session: str, timestamps: list[float]) -> Path:
    output = directory / session
    output.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "video": f"{session}.mp4",
        "model_version": "test-model",
        "event_count": len(timestamps),
        "events": [
            {
                "event_id": index + 1,
                "timestamp": value,
                "confidence": 0.5,
                "route": "pickup_to_customer",
                "evidence_file": f"events/event-{index + 1:04d}-{value:.3f}s.jpg",
            }
            for index, value in enumerate(timestamps)
        ],
    }
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_truth(path: Path, sessions: dict[str, list[dict[str, object]]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sessions": {
                    name: {"video": f"{name}.mp4", "transactions": rows}
                    for name, rows in sessions.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


class HandoverEvaluationTests(unittest.TestCase):
    def test_session_name_comes_from_the_replay_output_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            report = _write_report(Path(temporary), "shop-01-morning", [10.0])
            session = load_replay_session(report)
            self.assertEqual(session.session, "shop-01-morning")
            self.assertEqual(session.model_version, "test-model")
            self.assertEqual(len(session.events), 1)

    def test_matching_reports_correct_extra_and_missed_counts(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            # 26.0 matches, 120.0 is an extra count, 300.0 was missed.
            report = _write_report(directory, "session-a", [26.4, 120.0])
            truth = _write_truth(
                directory / "truth.json",
                {"session-a": [{"timestamp": 26.0}, {"timestamp": 300.0}]},
            )
            result = evaluate_handovers([report], truth, tolerance_seconds=7.0)
            overall = result["overall"]
            self.assertEqual(overall["true_positives"], 1)
            self.assertEqual(overall["false_positives"], 1)
            self.assertEqual(overall["false_negatives"], 1)
            self.assertEqual(overall["precision"], 0.5)
            self.assertEqual(overall["recall"], 0.5)

            score = result["sessions"][0]
            self.assertEqual(score["missed_truth_timestamps"], [300.0])
            self.assertEqual(score["extra_event_timestamps"], [120.0])
            self.assertAlmostEqual(score["matches"][0]["time_error_seconds"], 0.4, places=3)
            self.assertTrue(any("missed handover" in item for item in score["review_actions"]))
            self.assertTrue(any("extra count" in item for item in score["review_actions"]))

    def test_quantity_gap_exposes_simultaneous_handover_undercount(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = _write_report(directory, "session-a", [26.0, 200.0])
            truth = _write_truth(
                directory / "truth.json",
                {
                    "session-a": [
                        {"timestamp": 26.0, "quantity": 1},
                        {"timestamp": 200.0, "quantity": 2, "note": "two cones together"},
                    ]
                },
            )
            result = evaluate_handovers([report], truth, tolerance_seconds=7.0)
            overall = result["overall"]
            # Every transaction is found, but one ice cream was never counted.
            self.assertEqual(overall["recall"], 1.0)
            self.assertEqual(overall["precision"], 1.0)
            self.assertEqual(overall["truth_quantity"], 3)
            self.assertEqual(overall["detected_events"], 2)
            self.assertEqual(overall["quantity_gap"], -1)
            self.assertTrue(
                any("quantity undercount" in item for item in result["sessions"][0]["review_actions"])
            )

    def test_each_truth_and_prediction_is_matched_at_most_once(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            # Two detections sit inside the tolerance of a single real handover.
            report = _write_report(directory, "session-a", [26.0, 27.0])
            truth = _write_truth(directory / "truth.json", {"session-a": [{"timestamp": 26.2}]})
            result = evaluate_handovers([report], truth, tolerance_seconds=7.0)
            overall = result["overall"]
            self.assertEqual(overall["true_positives"], 1)
            self.assertEqual(overall["false_positives"], 1)
            self.assertEqual(overall["false_negatives"], 0)
            self.assertEqual(result["sessions"][0]["extra_event_timestamps"], [27.0])

    def test_tolerance_decides_whether_a_late_detection_counts(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = _write_report(directory, "session-a", [31.0])
            truth = _write_truth(directory / "truth.json", {"session-a": [{"timestamp": 26.0}]})
            tight = evaluate_handovers([report], truth, tolerance_seconds=2.0)
            loose = evaluate_handovers([report], truth, tolerance_seconds=7.0)
            self.assertEqual(tight["overall"]["true_positives"], 0)
            self.assertEqual(loose["overall"]["true_positives"], 1)

    def test_multiple_sessions_are_aggregated(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = _write_report(directory, "session-a", [26.0])
            second = _write_report(directory, "session-b", [10.0, 50.0])
            truth = _write_truth(
                directory / "truth.json",
                {
                    "session-a": [{"timestamp": 26.0}],
                    "session-b": [{"timestamp": 10.0}, {"timestamp": 50.0}],
                },
            )
            result = evaluate_handovers([first, second], truth, tolerance_seconds=7.0)
            self.assertEqual(result["overall"]["sessions"], 2)
            self.assertEqual(result["overall"]["true_positives"], 3)
            self.assertEqual(result["overall"]["recall"], 1.0)
            self.assertEqual([item["session"] for item in result["sessions"]], ["session-a", "session-b"])
            self.assertIn("precision / recall / f1", format_summary(result))

    def test_missing_truth_session_is_a_clear_error(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = _write_report(directory, "session-a", [26.0])
            truth = _write_truth(directory / "truth.json", {"session-other": [{"timestamp": 1.0}]})
            with self.assertRaises(HandoverEvaluationError) as context:
                evaluate_handovers([report], truth, tolerance_seconds=7.0)
            self.assertIn("session-a", str(context.exception))

    def test_zero_quantity_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = _write_report(directory, "session-a", [26.0])
            truth = _write_truth(
                directory / "truth.json", {"session-a": [{"timestamp": 26.0, "quantity": 0}]}
            )
            with self.assertRaises(HandoverEvaluationError):
                evaluate_handovers([report], truth, tolerance_seconds=7.0)

    def test_template_lists_every_detection_as_a_candidate_row(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = _write_report(directory, "session-a", [26.0, 90.0])
            second = _write_report(directory, "session-b", [10.0])
            template = build_truth_template([first, second])
            self.assertEqual(sorted(template["sessions"]), ["session-a", "session-b"])
            rows = template["sessions"]["session-a"]["transactions"]
            self.assertEqual([row["timestamp"] for row in rows], [26.0, 90.0])
            self.assertTrue(all(row["quantity"] == 1 for row in rows))

            # A filled-in template must be loadable straight back for scoring.
            truth_path = directory / "truth.json"
            truth_path.write_text(json.dumps(template, indent=2, sort_keys=True), encoding="utf-8")
            result = evaluate_handovers([first, second], truth_path, tolerance_seconds=1.0)
            self.assertEqual(result["overall"]["recall"], 1.0)
            self.assertEqual(result["overall"]["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
