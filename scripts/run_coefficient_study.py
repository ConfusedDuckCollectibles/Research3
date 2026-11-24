"""Supervise the one preregistered coefficient/TCP study and retain partial evidence."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "checkpoints/coefficient-packet"
BUDGET_SECONDS = 300
EXPECTED_TRIALS = 720
DESIGN_COMMIT = "49cfba7261326d58e174e51c8ddf5abba6f4a966"
DESIGN_PATH = "docs/coefficient_packet_design.md"
DESIGN_SHA256 = "b8307677f5320c3413b5a1007bdfe26230a844c57503d53b4443e5fc542647ba"
FROZEN_DEPENDENCIES = {
    "latent_bridge/alignment_core.py": "de8909692a928266c03e22d12b13c800b14e2495a8acad39ae6b95dfc46bfa53",
    "latent_bridge/alignment_packet.py": "6ae2c169b843007dd9793894676d8ce4a933c5f82fd2ce4063b83dea389c74fd",
    "scripts/run_alignment_numerical.py": "40f806d39f24a721009d4d7599c8902f1b6720956065daa82e0f4b747ab7a4d4",
    "docs/alignment_geometry.md": "06fa7747aa96db8719ce7ae930789daee6291c65cab6752fcadb1364f0f6f2ff",
}
SOURCE_PATHS = (
    DESIGN_PATH, *FROZEN_DEPENDENCIES,
    "latent_bridge/__init__.py",
    "latent_bridge/coefficient_packet.py",
    "latent_bridge/coefficient_transport.py",
    "latent_bridge/coefficient_study.py",
    "scripts/run_coefficient_study.py",
    "tests/test_coefficient_packet.py",
    "tests/test_coefficient_transport.py",
    "tests/test_coefficient_execution.py",
    "tests/test_alignment_execution.py",
    ".github/workflows/coefficient-packet-v2.yml",
    "pyproject.toml",
)

# Reuse the already-tested immutable process-group deadline and gzip preservation
# helpers. Importing this module does not call its original-study main function.
_SPEC = importlib.util.spec_from_file_location(
    "frozen_alignment_supervision", ROOT / "scripts/run_alignment_numerical.py")
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)
info = _HELPERS.info
write_json = _HELPERS.write_json
run_child = _HELPERS.run_child
preserve_gzip = _HELPERS.preserve_gzip


def require_context() -> dict:
    env = os.environ
    if (env.get("GITHUB_ACTIONS") != "true"
            or env.get("GITHUB_REPOSITORY") != "ConfusedDuckCollectibles/Research3"
            or env.get("GITHUB_REF") != "refs/heads/main"
            or env.get("GITHUB_RUN_ATTEMPT") != "1"
            or env.get("GITHUB_RUN_NUMBER") != "1"):
        raise RuntimeError("Only the first original coefficient-v2 main workflow run is permitted")
    if platform.python_version() != "3.11.18":
        raise RuntimeError("Frozen Python runtime mismatch")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()
    if head != env.get("GITHUB_SHA"):
        raise RuntimeError("Checkout and workflow source differ")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError("Refusing to overwrite existing coefficient output")
    if (ROOT / "results/coefficient_packet").exists():
        raise RuntimeError("Permanent coefficient results already exist; do not repeat")
    if info(ROOT / DESIGN_PATH)["sha256"] != DESIGN_SHA256:
        raise RuntimeError("Frozen coefficient design changed")
    for path, expected in FROZEN_DEPENDENCIES.items():
        if info(ROOT / path)["sha256"] != expected:
            raise RuntimeError("Frozen original dependency changed: " + path)
    return {
        "github_commit": head, "github_run_id": env["GITHUB_RUN_ID"],
        "github_run_number": 1, "github_run_attempt": 1,
        "frozen_design_commit": DESIGN_COMMIT, "design_sha256": DESIGN_SHA256,
    }


def completion_accounting(directory: Path) -> dict:
    """Account for every planned slot even if a hard kill interrupts a JSONL row.

    No study imports, fixture generation, packet replay, or timing takes place.
    Raw files are never repaired or rewritten by this reader.
    """
    result = {
        "expected_trials": EXPECTED_TRIALS, "plan_status": "unavailable",
        "planned_slots": 0, "recorded_slots": 0, "unique_recorded_slots": 0,
        "status_counts": {}, "duplicate_ids": [], "unknown_ids": [],
        "read_errors": [], "unexecuted_slots": [],
        "all_slots_recorded": False, "all_trials_completed_successfully": False,
    }
    plan_path = directory / "plan.json"
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        rows = plan["planned_trials"]
        if (plan.get("schema_version") != 1
                or plan.get("expected_count") != EXPECTED_TRIALS
                or not isinstance(rows, list) or len(rows) != EXPECTED_TRIALS):
            raise ValueError("plan version/count/shape differs from the frozen grid")
        ids = [row["trial_id"] for row in rows]
        if (any(not isinstance(i, str) or not i for i in ids)
                or len(set(ids)) != EXPECTED_TRIALS):
            raise ValueError("plan has invalid or repeated trial IDs")
        result["plan_status"] = "valid"
        result["planned_slots"] = len(rows)
    except Exception as error:
        result["read_errors"].append({
            "file": "plan.json", "type": type(error).__name__, "message": str(error)})
        return result
    planned = set(ids)
    seen = Counter()
    statuses = Counter()
    invalid_outcomes = []
    trial_path = directory / "trials.jsonl"
    if trial_path.exists():
        with trial_path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    result["read_errors"].append(
                        {"file": "trials.jsonl", "line": line_number,
                         "type": "EmptyRecord"})
                    continue
                try:
                    row = json.loads(line)
                    identifier = row["trial_id"]
                    if not isinstance(identifier, str):
                        raise ValueError("trial ID is not a string")
                    status = row["status"]
                    if status not in {"completed", "error", "numerically_ineligible",
                                      "not_executed"}:
                        raise ValueError("unknown trial outcome")
                    wall = row.get("wall_ns")
                    if status == "completed" and (isinstance(wall, bool)
                            or not isinstance(wall, int) or wall < 0):
                        invalid_outcomes.append(identifier)
                    seen[identifier] += 1
                    statuses[status] += 1
                except Exception as error:
                    result["read_errors"].append(
                        {"file": "trials.jsonl", "line": line_number,
                         "type": type(error).__name__, "message": str(error),
                         "ended_with_newline": line.endswith(b"\n")})
    result["recorded_slots"] = sum(seen.values())
    result["unique_recorded_slots"] = len(seen)
    result["status_counts"] = dict(sorted(statuses.items()))
    result["duplicate_ids"] = sorted(i for i, n in seen.items() if n > 1)
    result["unknown_ids"] = sorted(set(seen) - planned)
    result["invalid_completed_wall_times"] = sorted(invalid_outcomes)
    result["unexecuted_slots"] = [
        {"trial_id": row["trial_id"], "status": "not_executed",
         "reason": "no complete retained trial record after child termination"}
        for row in rows if row["trial_id"] not in seen
    ]
    result["all_slots_recorded"] = (
        set(seen) == planned and not result["duplicate_ids"]
        and not result["unknown_ids"] and not result["read_errors"])
    result["all_trials_completed_successfully"] = (
        result["all_slots_recorded"] and not invalid_outcomes
        and statuses == Counter({"completed": EXPECTED_TRIALS}))
    return result


def emit_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size > 300_000:
        return
    raw = path.read_bytes()
    raw.decode("utf-8")
    print("COEFFICIENT_FILE_BEGIN " + path.name + " "
          + hashlib.sha256(raw).hexdigest() + " " + str(len(raw)), flush=True)
    sys.stdout.buffer.write(raw)
    if not raw.endswith(b"\n"):
        sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()
    print("COEFFICIENT_FILE_END " + path.name, flush=True)


def main() -> None:
    identity = require_context()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = {name: info(ROOT / name) for name in SOURCE_PATHS}
    write_json(OUTPUT / "launch_source_manifest.json",
               {**identity, "files": files, "model_downloads": 0})
    environment = dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "latent_bridge.coefficient_study",
               "--output", str(OUTPUT)]
    write_json(OUTPUT / "launcher.json", {
        **identity, "command": command, "budget_seconds": BUDGET_SECONDS,
        "scope": "fixed_synthetic_numerical_and_loopback_TCP_no_models",
        "expected_network_trials": EXPECTED_TRIALS,
        "network_scope": "127.0.0.1 only; imposed application pacing/holdoff",
        "threads": {name: environment[name] for name in
                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "artifact_retention_days": 90,
    })
    outcome = run_child(command, OUTPUT / "worker.log", environment, BUDGET_SECONDS)
    retention_started = time.perf_counter()
    accounting = completion_accounting(OUTPUT)
    write_json(OUTPUT / "completion_accounting.json", accounting)
    compression = {}
    retention_errors = []
    for path in sorted(OUTPUT.rglob("*.jsonl")):
        if path.is_symlink():
            retention_errors.append(
                {"file": path.relative_to(OUTPUT).as_posix(), "type": "UnexpectedSymlink"})
            continue
        try:
            compression[path.relative_to(OUTPUT).as_posix()] = preserve_gzip(path)
        except Exception as error:
            retention_errors.append(
                {"file": path.relative_to(OUTPUT).as_posix(),
                 "type": type(error).__name__, "message": str(error)})
    inventory = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_symlink():
            retention_errors.append(
                {"file": path.relative_to(OUTPUT).as_posix(), "type": "UnexpectedSymlink"})
        elif path.is_file() and path.name != "execution.json":
            inventory[path.relative_to(OUTPUT).as_posix()] = info(path)
    execution = {
        **identity, **outcome, "budget_seconds": BUDGET_SECONDS,
        "completion_rule": (
            "All frozen slots and required checks must pass for complete scientific "
            "verification. A negative performance gate is a valid completed result."),
        "run_kind": "preregistered_coefficient_packet_v2",
        "artifact_retention_days": 90, "compression_receipts": compression,
        "retention_errors": retention_errors, "retained_files": inventory,
        "all_slots_recorded": accounting["all_slots_recorded"],
        "all_trials_completed_successfully": accounting["all_trials_completed_successfully"],
        "benchmark_retries": 0, "model_imports_or_inference": 0,
        "supervisor_retention_wall_seconds": time.perf_counter() - retention_started,
        "retention_budget_scope": (
            "Post-child byte-preserving packaging and missing-slot accounting; "
            "no scientific work is resumed or added after the 300-second child deadline."),
    }
    write_json(OUTPUT / "execution.json", execution)
    for name in ("results.json", "report.md", "status.json", "runtime_manifest.json",
                 "worker_failure.json", "completion_accounting.json",
                 "worker.log", "execution.json"):
        emit_file(OUTPUT / name)
    print(json.dumps({
        "worker_returncode": outcome["worker_returncode"],
        "failure": outcome["failure"], "retention_errors": retention_errors,
        "retained_files": len(inventory),
        "child_wall_seconds": outcome["child_wall_seconds_including_imports"],
        "all_slots_recorded": accounting["all_slots_recorded"],
        "all_trials_completed_successfully": accounting["all_trials_completed_successfully"],
    }, sort_keys=True), flush=True)
    if (outcome["worker_returncode"] != 0 or outcome["failure"] or retention_errors
            or not accounting["all_trials_completed_successfully"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
