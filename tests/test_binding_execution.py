"""Exercise the real wrapper's timeout/retention path with harmless subprocesses.

No pretrained weights, network calls, fits, or study outcomes are used here.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("binding_execution_wrapper", SCRIPTS / "run_binding_study.py")
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)

FIRST_RUN = {
    "GITHUB_ACTIONS": "true", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_RUN_NUMBER": "1",
    "GITHUB_REF": "refs/heads/main", "GITHUB_SHA": "unit-test-only", "GITHUB_RUN_ID": "0",
}


class BindingExecutionGuardTests(unittest.TestCase):
    def test_rejects_retries_other_refs_existing_work_and_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "checkpoints" / "binding-study"
            with mock.patch.object(WRAPPER, "ROOT", root), mock.patch.object(WRAPPER, "OUTPUT", output):
                with mock.patch.dict(os.environ, FIRST_RUN):
                    WRAPPER.require_first_attempt()
                changes = (
                    {"GITHUB_ACTIONS": "false"}, {"GITHUB_RUN_ATTEMPT": "2"},
                    {"GITHUB_RUN_NUMBER": "2"}, {"GITHUB_REF": "refs/heads/other"},
                )
                for change in changes:
                    with self.subTest(change=change), mock.patch.dict(os.environ, {**FIRST_RUN, **change}):
                        with self.assertRaises(RuntimeError):
                            WRAPPER.require_first_attempt()
                with mock.patch.dict(os.environ, FIRST_RUN):
                    output.mkdir(parents=True)
                    (output / "partial.json").write_text("{}", encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        WRAPPER.require_first_attempt()
                    (output / "partial.json").unlink()
                    (root / "results" / "binding_study").mkdir(parents=True)
                    with self.assertRaises(RuntimeError):
                        WRAPPER.require_first_attempt()

    @unittest.skipUnless(os.name == "posix" and Path("/proc").exists(), "Linux process-group contract")
    def test_global_timeout_kills_descendants_and_preserves_partial_prediction_bytes(self):
        original_popen = subprocess.Popen
        process_ids = {}
        raw = b'{"completed":true}\n{"interrupted":'
        child_code = (
            "import json,os,subprocess,sys,time\n"
            "from pathlib import Path\n"
            "out=Path(sys.argv[1])\n"
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
            "(out/'unit_processes.json').write_text(json.dumps({'parent':os.getpid(),'child':child.pid}))\n"
            "(out/'predictions.jsonl').write_bytes(b'{\"completed\":true}\\n{\"interrupted\":')\n"
            "(out/'status.json').write_text(json.dumps({'status':'unit_partial'}))\n"
            "time.sleep(60)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "checkpoints" / "binding-study"
            (root / "docs").mkdir()
            (root / "docs" / "binding_design.md").write_text("unit fixture only\n", encoding="utf-8")

            def launch(_command, **kwargs):
                self.assertTrue(kwargs["start_new_session"])
                return original_popen([sys.executable, "-c", child_code, str(output)], **kwargs)

            try:
                with (
                    mock.patch.object(WRAPPER, "ROOT", root),
                    mock.patch.object(WRAPPER, "OUTPUT", output),
                    mock.patch.object(WRAPPER, "BUDGET_SECONDS", 1.0),
                    mock.patch.object(WRAPPER.subprocess, "Popen", side_effect=launch),
                    mock.patch.dict(os.environ, FIRST_RUN),
                ):
                    with self.assertRaises(SystemExit) as raised:
                        WRAPPER.main()
                self.assertEqual(raised.exception.code, 1)
                execution = json.loads((output / "execution.json").read_text(encoding="utf-8"))
                self.assertEqual(execution["failure"]["type"], "GlobalStudyTimeout")
                self.assertEqual(execution["study_status"]["status"], "unit_partial")
                self.assertLess(execution["study_wall_seconds_including_downloads"], 5.0)
                self.assertEqual((output / "predictions.jsonl").read_bytes(), raw)
                self.assertEqual(gzip.decompress((output / "predictions.jsonl.gz").read_bytes()), raw)
                self.assertTrue(execution["prediction_receipts"]["predictions.jsonl"]["round_trip_verified"])
                self.assertIn("worker.log", execution["retained_files"])
                process_ids = json.loads((output / "unit_processes.json").read_text(encoding="utf-8"))
                for label, pid in process_ids.items():
                    state = Path(f"/proc/{pid}/stat")
                    deadline = time.monotonic() + 1.0
                    while state.exists() and time.monotonic() < deadline:
                        try:
                            # A killed orphan may remain briefly as a zombie
                            # until the runner's init reaps it; it cannot execute.
                            if state.read_text().split(") ", 1)[1].split()[0] == "Z":
                                break
                        except FileNotFoundError:
                            break
                        time.sleep(0.02)
                    try:
                        final_state = state.read_text().split(") ", 1)[1].split()[0]
                    except FileNotFoundError:
                        final_state = None
                    if final_state is not None:
                        self.assertEqual(final_state, "Z", label)
            finally:
                ids_path = output / "unit_processes.json"
                if ids_path.exists():
                    process_ids = json.loads(ids_path.read_text())
                if process_ids.get("parent"):
                    try:
                        os.killpg(process_ids["parent"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
