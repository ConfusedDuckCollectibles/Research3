"""Operational contracts specific to complete/partial coefficient evidence."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "coefficient_execution_wrapper", ROOT / "scripts/run_coefficient_study.py")
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


def write_plan(directory):
    rows = [{"trial_id": "synthetic-retention-slot-" + str(i)} for i in range(720)]
    (directory / "plan.json").write_text(
        json.dumps({"schema_version": 1, "expected_count": 720, "planned_trials": rows}),
        encoding="utf-8")
    return rows


class CoefficientExecutionTests(unittest.TestCase):
    def test_partial_record_stays_untouched_and_every_missing_slot_is_named(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = write_plan(root)
            raw = (json.dumps({"trial_id": plan[0]["trial_id"], "status": "completed",
                               "wall_ns": 123}) + "\n").encode() + b'{"trial_id":'
            path = root / "trials.jsonl"
            path.write_bytes(raw)
            result = WRAPPER.completion_accounting(root)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(result["recorded_slots"], 1)
            self.assertEqual(len(result["unexecuted_slots"]), 719)
            self.assertEqual(result["unexecuted_slots"][0]["trial_id"], plan[1]["trial_id"])
            self.assertFalse(result["all_slots_recorded"])
            self.assertFalse(result["all_trials_completed_successfully"])
            self.assertEqual(result["read_errors"][0]["line"], 2)

    def test_failed_duplicate_and_unknown_records_cannot_claim_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = write_plan(root)
            rows = [{"trial_id": r["trial_id"], "status": "completed", "wall_ns": 7}
                    for r in plan]
            rows[3]["status"] = "error"
            rows[3]["wall_ns"] = None
            rows.append(dict(rows[0]))
            rows.append({"trial_id": "not-in-plan", "status": "completed", "wall_ns": 8})
            (root / "trials.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            result = WRAPPER.completion_accounting(root)
            self.assertEqual(result["duplicate_ids"], [plan[0]["trial_id"]])
            self.assertEqual(result["unknown_ids"], ["not-in-plan"])
            self.assertFalse(result["all_slots_recorded"])
            self.assertFalse(result["all_trials_completed_successfully"])
            self.assertEqual(result["status_counts"]["error"], 1)
            self.assertEqual(result["unexecuted_slots"], [])

    def test_complete_exact_grid_is_distinct_from_missing_or_invalid_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            absent = WRAPPER.completion_accounting(root)
            self.assertEqual(absent["plan_status"], "unavailable")
            self.assertFalse(absent["all_trials_completed_successfully"])
            plan = write_plan(root)
            (root / "trials.jsonl").write_text(
                "".join(json.dumps({"trial_id": r["trial_id"],
                                     "status": "completed", "wall_ns": 1}) + "\n"
                        for r in plan), encoding="utf-8")
            full = WRAPPER.completion_accounting(root)
            self.assertTrue(full["all_slots_recorded"])
            self.assertTrue(full["all_trials_completed_successfully"])
            self.assertEqual(full["unexecuted_slots"], [])

    def test_original_run_guard_rejects_retry_before_git_or_execution(self):
        context = {"GITHUB_ACTIONS": "true",
                   "GITHUB_REPOSITORY": "ConfusedDuckCollectibles/Research3",
                   "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_NUMBER": "1",
                   "GITHUB_RUN_ATTEMPT": "2"}
        with patch.dict(os.environ, context, clear=True):
            with patch.object(WRAPPER.subprocess, "run") as git:
                with self.assertRaisesRegex(RuntimeError, "first original"):
                    WRAPPER.require_context()
                git.assert_not_called()


if __name__ == "__main__":
    unittest.main()
