"""Operational checks for deadline supervision and byte-preserving retention."""
from __future__ import annotations
import gzip
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "alignment_execution_wrapper", ROOT / "scripts/run_alignment_numerical.py")
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


class AlignmentExecutionTests(unittest.TestCase):
    def test_successful_child_output_and_status_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "worker.log"
            result = WRAPPER.run_child(
                [sys.executable, "-c", "print('completed synthetic supervision fixture')"],
                log, dict(os.environ), 3.0)
            self.assertEqual(result["worker_returncode"], 0)
            self.assertIsNone(result["failure"])
            self.assertIn(b"completed synthetic supervision fixture", log.read_bytes())

    @unittest.skipUnless(hasattr(os, "killpg"), "process-group supervision requires POSIX")
    def test_timeout_kills_descendant_before_delayed_side_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready, sentinel, log = root / "ready", root / "delayed", root / "worker.log"
            grandchild = ("import pathlib,sys,time; time.sleep(1.0); "
                          "pathlib.Path(sys.argv[1]).write_text('should not happen')")
            child = ("import pathlib,subprocess,sys,time; "
                     "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[3]]); "
                     "pathlib.Path(sys.argv[2]).write_text('spawned'); "
                     "print('fixture ready',flush=True); time.sleep(30)")
            result = WRAPPER.run_child(
                [sys.executable, "-c", child, grandchild, str(ready), str(sentinel)],
                log, dict(os.environ), 0.6)
            self.assertTrue(ready.is_file(), "the descendant was actually launched")
            self.assertEqual(result["failure"]["type"], "NumericalStudyTimeout")
            self.assertNotEqual(result["worker_returncode"], 0)
            self.assertLess(result["child_wall_seconds_including_imports"], 4.0)
            time.sleep(0.7)
            self.assertFalse(sentinel.exists(), "descendant survived the process-group timeout")
            self.assertIn(b"fixture ready", log.read_bytes())

    def test_existing_valid_gzip_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "cases.jsonl"
            raw.write_bytes(b'{"synthetic_retention_fixture":true}\n')
            compressed = raw.with_name(raw.name + ".gz")
            previous = gzip.compress(raw.read_bytes(), mtime=123)
            compressed.write_bytes(previous)
            receipt = WRAPPER.preserve_gzip(raw)
            self.assertEqual(compressed.read_bytes(), previous)
            self.assertTrue(receipt["round_trip_verified"])
            self.assertEqual(receipt["raw"], WRAPPER.info(raw))
            self.assertEqual(receipt["gzip"], WRAPPER.info(compressed))

    def test_contradictory_gzip_is_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "timings.jsonl"
            original = b'{"duration_ns":123}\n'
            raw.write_bytes(original)
            compressed = raw.with_name(raw.name + ".gz")
            contradiction = gzip.compress(b'{"duration_ns":999}\n', mtime=123)
            compressed.write_bytes(contradiction)
            with self.assertRaises(RuntimeError):
                WRAPPER.preserve_gzip(raw)
            self.assertEqual(raw.read_bytes(), original)
            self.assertEqual(compressed.read_bytes(), contradiction)


if __name__ == "__main__":
    unittest.main()
