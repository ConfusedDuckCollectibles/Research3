"""Independent operational accounting controls; no models or study outcomes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "contextual_supervision_contract", ROOT / "scripts/run_contextual_information.py"
)
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def expected_ids():
    # Specified order, independently constructed without the worker/generator.
    return [
        f"F{family:03d}/{variant}/{probe}"
        for family in range(64)
        for variant in ("original", "swapped")
        for probe in ("unique", "repeated")
    ]


class ContextualExecutionContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write_plan(self, ids=None, expected_count=256):
        ids = expected_ids() if ids is None else ids
        plan = {
            "schema_version": 1,
            "expected_count": expected_count,
            "planned_records": [{"record_id": identifier} for identifier in ids],
        }
        (self.directory / "plan.json").write_text(
            json.dumps(plan) + "\n", encoding="utf-8"
        )

    def write_records(self, rows, tail=b""):
        raw = b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
            for row in rows
        ) + tail
        path = self.directory / "records.jsonl"
        path.write_bytes(raw)
        return path, raw

    def test_complete_negative_recovery_is_complete_evidence(self):
        self.write_plan()
        rows = [
            {
                "record_id": identifier,
                "status": "completed",
                "inverse": {"status": "rejected", "reason": "spectral_domain"},
            }
            for identifier in expected_ids()
        ]
        path, original = self.write_records(rows)
        (self.directory / "results.json").write_text(
            '{"primary_recovery_target_passed":false}\n', encoding="utf-8"
        )
        result = supervisor.completion_accounting(self.directory)
        self.assertEqual(result["expected_records"], 256)
        self.assertEqual(result["planned_slots"], 256)
        self.assertEqual(result["recorded_slots"], 256)
        self.assertEqual(result["unique_recorded_slots"], 256)
        self.assertEqual(result["status_counts"], {"completed": 256})
        self.assertEqual(result["unexecuted_slots"], [])
        self.assertEqual(result["read_errors"], [])
        self.assertTrue(result["all_slots_recorded"])
        self.assertTrue(result["all_records_completed_successfully"])
        self.assertEqual(path.read_bytes(), original)

    def test_torn_tail_is_retained_and_missing_slots_are_not_successes(self):
        self.write_plan()
        path, original = self.write_records(
            [{"record_id": expected_ids()[0], "status": "completed"}],
            b'{"record_id":"F000/original/repeated","status":',
        )
        result = supervisor.completion_accounting(self.directory)
        self.assertEqual(result["recorded_slots"], 1)
        self.assertEqual(result["status_counts"], {"completed": 1})
        self.assertEqual(len(result["unexecuted_slots"]), 255)
        self.assertEqual(result["unexecuted_slots"][0]["record_id"], expected_ids()[1])
        self.assertEqual(result["unexecuted_slots"][-1]["record_id"], expected_ids()[-1])
        self.assertEqual(len(result["read_errors"]), 1)
        self.assertEqual(result["read_errors"][0]["line"], 2)
        self.assertFalse(result["read_errors"][0]["ended_with_newline"])
        self.assertFalse(result["all_slots_recorded"])
        self.assertFalse(result["all_records_completed_successfully"])
        self.assertEqual(path.read_bytes(), original)

    def test_duplicates_and_unknown_records_cannot_replace_the_fixed_cohort(self):
        self.write_plan()
        rows = [
            {"record_id": identifier, "status": "completed"}
            for identifier in expected_ids()
        ]
        rows += [
            {"record_id": expected_ids()[0], "status": "completed"},
            {"record_id": "F999/original/unique", "status": "completed"},
        ]
        path, original = self.write_records(rows)
        result = supervisor.completion_accounting(self.directory)
        self.assertEqual(result["recorded_slots"], 258)
        self.assertEqual(result["unique_recorded_slots"], 257)
        self.assertEqual(result["duplicate_ids"], [expected_ids()[0]])
        self.assertEqual(result["unknown_ids"], ["F999/original/unique"])
        self.assertEqual(result["unexecuted_slots"], [])
        self.assertFalse(result["all_slots_recorded"])
        self.assertFalse(result["all_records_completed_successfully"])
        self.assertEqual(path.read_bytes(), original)

    def test_recorded_errors_and_not_executed_are_not_completed(self):
        self.write_plan()
        rows = [
            {"record_id": identifier, "status": "completed"}
            for identifier in expected_ids()
        ]
        rows[0]["status"] = "error"
        rows[1]["status"] = "not_executed"
        self.write_records(rows)
        result = supervisor.completion_accounting(self.directory)
        self.assertTrue(result["all_slots_recorded"])
        self.assertFalse(result["all_records_completed_successfully"])
        self.assertEqual(result["unexecuted_slots"], [])
        self.assertEqual(
            result["status_counts"], {"completed": 254, "error": 1, "not_executed": 1}
        )

    def test_plan_count_and_order_must_match_before_accounting(self):
        for alteration in ("swapped_order", "wrong_count", "replacement_id"):
            with self.subTest(alteration=alteration):
                ids = expected_ids()
                if alteration == "swapped_order":
                    ids[1], ids[2] = ids[2], ids[1]
                elif alteration == "replacement_id":
                    ids[-1] = "F999/swapped/repeated"
                self.write_plan(ids, 255 if alteration == "wrong_count" else 256)
                self.write_records(
                    {"record_id": identifier, "status": "completed"}
                    for identifier in expected_ids()
                )
                result = supervisor.completion_accounting(self.directory)
                self.assertEqual(result["plan_status"], "unavailable")
                self.assertEqual(result["recorded_slots"], 0)
                self.assertEqual(result["read_errors"][0]["file"], "plan.json")
                self.assertFalse(result["all_slots_recorded"])

    def test_absence_and_malformed_rows_remain_distinct(self):
        self.write_plan()
        result = supervisor.completion_accounting(self.directory)
        self.assertEqual(result["recorded_slots"], 0)
        self.assertEqual(len(result["unexecuted_slots"]), 256)
        self.assertEqual(result["read_errors"], [])

        path, original = self.write_records(
            [
                {"record_id": expected_ids()[0], "status": "completed"},
                {"record_id": expected_ids()[1], "status": "recovered"},
            ],
            b"\nnot-json\n",
        )
        result = supervisor.completion_accounting(self.directory)
        self.assertEqual(result["recorded_slots"], 1)
        self.assertEqual(len(result["unexecuted_slots"]), 255)
        self.assertEqual(len(result["read_errors"]), 3)
        self.assertTrue(
            any(error["type"] == "EmptyRecord" for error in result["read_errors"])
        )
        self.assertFalse(result["all_records_completed_successfully"])
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
