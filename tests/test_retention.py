"""Tests for plan-first bounded evidence retention."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.operations.retention import RetentionManager, RetentionPolicy  # noqa: E402


class RetentionManagerTests(unittest.TestCase):
    def test_dry_run_never_deletes_and_real_execution_deletes_only_planned_media(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "session" / "old.jpg"
            fresh = root / "session" / "fresh.jpg"
            report = root / "session" / "report.json"
            old.parent.mkdir()
            old.write_bytes(b"old")
            fresh.write_bytes(b"fresh")
            report.write_text("keep", encoding="utf-8")
            old_timestamp = datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()
            os.utime(old, (old_timestamp, old_timestamp))
            now = datetime(2026, 8, 5, tzinfo=timezone.utc)
            manager = RetentionManager(root, RetentionPolicy(max_age=timedelta(days=7)))

            plan = manager.plan(now=now)
            self.assertEqual([item.relative_path for item in plan.candidates], ["session/old.jpg"])
            result = manager.execute(plan)
            self.assertTrue(result.dry_run)
            self.assertTrue(old.exists())

            result = manager.execute(plan, dry_run=False)
            self.assertEqual(result.deleted_files, 1)
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(report.exists())

    def test_changed_file_is_skipped_from_stale_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "old.mp4"
            candidate.write_bytes(b"old")
            old_timestamp = datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()
            os.utime(candidate, (old_timestamp, old_timestamp))
            manager = RetentionManager(root, RetentionPolicy(max_age=timedelta(days=7)))
            plan = manager.plan(now=datetime(2026, 8, 5, tzinfo=timezone.utc))
            candidate.write_bytes(b"new-content")

            result = manager.execute(plan, dry_run=False)
            self.assertEqual(result.deleted_files, 0)
            self.assertEqual(result.skipped, ("old.mp4",))
            self.assertTrue(candidate.exists())

    def test_quota_selects_oldest_files_until_within_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(3):
                path = root / f"{index}.jpg"
                path.write_bytes(b"x" * 10)
                os.utime(path, (index + 1, index + 1))
            manager = RetentionManager(root, RetentionPolicy(max_total_bytes=15))
            plan = manager.plan(now=datetime(2026, 8, 5, tzinfo=timezone.utc))
            self.assertEqual([item.relative_path for item in plan.candidates], ["0.jpg", "1.jpg"])
            self.assertEqual(plan.reclaimable_bytes, 20)


if __name__ == "__main__":
    unittest.main()
