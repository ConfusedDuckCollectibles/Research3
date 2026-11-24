"""Supervise the one frozen contextual-information assay and retain partial evidence.

The parent performs source checks, deadline enforcement, byte-preserving
packaging and record accounting. It never loads a model or resumes the assay.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "checkpoints/contextual-information"
BUDGET_SECONDS = 600
EXPECTED_RECORDS = 256
DESIGN_COMMIT = "c23a20ec420ba19680801a09ac8b053735807d38"
DESIGN_PATH = "docs/contextual_information_design.md"
DESIGN_SHA256 = "6f50e1d3faad3240033748d7434f088ec26b7ac6716511923892e66ac0e78a09"
FROZEN_DEPENDENCIES = {
    "latent_bridge/alignment_core.py": "de8909692a928266c03e22d12b13c800b14e2495a8acad39ae6b95dfc46bfa53",
    "scripts/run_alignment_numerical.py": "40f806d39f24a721009d4d7599c8902f1b6720956065daa82e0f4b747ab7a4d4",
    "docs/statebridge_information_recovery.md": "f78a41cee2b1ee90b349bcbee0fe00a588b9738863034172a8c8c09ddc3992d7",
    "third_party/statebridge/state_bridge.py": "415957df427eb79a4212e760af4d133cb4ce10cc4619f11447082117922f366d",
}
SOURCE_PATHS = (
    DESIGN_PATH, *FROZEN_DEPENDENCIES,
    "latent_bridge/__init__.py",
    "latent_bridge/information_recovery.py",
    "latent_bridge/contextual_information.py",
    "scripts/run_contextual_information.py",
    "scripts/contextual_model_lock.json",
    "third_party/statebridge/LICENSE",
    "third_party/statebridge/README.md",
    "tests/test_information_recovery.py",
    "tests/test_contextual_information.py",
    "tests/test_contextual_execution.py",
    "tests/test_alignment_execution.py",
    ".github/workflows/contextual-information-v1.yml",
    ".github/workflows/tests.yml",
    "pyproject.toml",
)

# The helper file is immutable and its whole-process-group deadline and exact
# gzip preservation have already been executed under repository tests.
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
        raise RuntimeError("Only the first original contextual-information main workflow run is permitted")
    if platform.python_version() != "3.11.18":
        raise RuntimeError("Frozen Python runtime mismatch")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()
    if head != env.get("GITHUB_SHA"):
        raise RuntimeError("Checkout and workflow source differ")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError("Refusing to overwrite existing contextual-information output")
    if (ROOT / "results/contextual_information").exists():
        raise RuntimeError("Permanent contextual-information results already exist; do not repeat")
    if info(ROOT / DESIGN_PATH)["sha256"] != DESIGN_SHA256:
        raise RuntimeError("Frozen contextual-information design changed")
    for path, expected in FROZEN_DEPENDENCIES.items():
        if info(ROOT / path)["sha256"] != expected:
            raise RuntimeError("Frozen dependency changed: " + path)
    if len(set(SOURCE_PATHS)) != len(SOURCE_PATHS):
        raise RuntimeError("Repeated source inventory path")
    return {
        "github_commit": head, "github_run_id": env["GITHUB_RUN_ID"],
        "github_run_number": 1, "github_run_attempt": 1,
        "frozen_design_commit": DESIGN_COMMIT, "design_sha256": DESIGN_SHA256,
    }


def fixed_record_ids() -> list[str]:
    """Operational slot identities only; no private-context generation."""
    return [f"F{f:03d}/{world}/{probe}" for f in range(64)
            for world in ("original", "swapped")
            for probe in ("unique", "repeated")]


def completion_accounting(directory: Path) -> dict:
    """Read retained rows, including interrupted raw files, without repair.

    An inverse rejection is a completed scientific observation. Operational
    errors and absent records are distinct. The parent never infers numerical
    success from an exit code or synthesizes an unobserved recovery.
    """
    fixed = fixed_record_ids()
    result = {
        "expected_records": EXPECTED_RECORDS, "plan_status": "unavailable",
        "planned_slots": 0, "recorded_slots": 0, "unique_recorded_slots": 0,
        "status_counts": {}, "duplicate_ids": [], "unknown_ids": [],
        "read_errors": [], "unexecuted_slots": [],
        "all_slots_recorded": False, "all_records_completed_successfully": False,
    }
    try:
        plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        rows = plan["planned_records"]
        if (plan.get("schema_version") != 1
                or plan.get("expected_count") != EXPECTED_RECORDS
                or not isinstance(rows, list) or len(rows) != EXPECTED_RECORDS):
            raise ValueError("plan version/count/shape differs from the frozen cohort")
        ids = [row["record_id"] for row in rows]
        if ids != fixed:
            raise ValueError("plan identity/order differs from the frozen cohort")
        result["plan_status"] = "valid"
        result["planned_slots"] = len(rows)
    except Exception as error:
        result["read_errors"].append({
            "file": "plan.json", "type": type(error).__name__, "message": str(error)})
        return result
    planned = set(ids)
    seen, statuses = Counter(), Counter()
    path = directory / "records.jsonl"
    if path.exists():
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    result["read_errors"].append(
                        {"file": "records.jsonl", "line": line_number, "type": "EmptyRecord"})
                    continue
                try:
                    row = json.loads(line)
                    identifier, status = row["record_id"], row["status"]
                    if not isinstance(identifier, str):
                        raise ValueError("record ID is not a string")
                    if status not in {"completed", "error", "not_executed"}:
                        raise ValueError("unknown record outcome")
                    seen[identifier] += 1
                    statuses[status] += 1
                except Exception as error:
                    result["read_errors"].append({
                        "file": "records.jsonl", "line": line_number,
                        "type": type(error).__name__, "message": str(error),
                        "ended_with_newline": line.endswith(b"\n")})
    result["recorded_slots"] = sum(seen.values())
    result["unique_recorded_slots"] = len(seen)
    result["status_counts"] = dict(sorted(statuses.items()))
    result["duplicate_ids"] = sorted(i for i, n in seen.items() if n > 1)
    result["unknown_ids"] = sorted(set(seen) - planned)
    result["unexecuted_slots"] = [
        {"record_id": identifier, "status": "not_executed",
         "reason": "no complete retained record after child termination"}
        for identifier in ids if identifier not in seen
    ]
    result["all_slots_recorded"] = (
        set(seen) == planned and not result["duplicate_ids"]
        and not result["unknown_ids"] and not result["read_errors"])
    result["all_records_completed_successfully"] = (
        result["all_slots_recorded"] and statuses == Counter({"completed": EXPECTED_RECORDS}))
    return result


def emit_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size > 300_000:
        return
    raw = path.read_bytes()
    raw.decode("utf-8")
    print("CONTEXTUAL_FILE_BEGIN " + path.name + " "
          + hashlib.sha256(raw).hexdigest() + " " + str(len(raw)), flush=True)
    sys.stdout.buffer.write(raw)
    if not raw.endswith(b"\n"):
        sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()
    print("CONTEXTUAL_FILE_END " + path.name, flush=True)


def main() -> None:
    identity = require_context()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = {name: info(ROOT / name) for name in SOURCE_PATHS}
    write_json(OUTPUT / "launch_source_manifest.json", {
        **identity, "files": files,
        "parent_model_imports_or_inference": 0,
        "model_download_scope": "Only the scientific child may download the six pinned public files.",
    })
    environment = dict(os.environ)
    environment.update(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    command = [sys.executable, "-m", "latent_bridge.contextual_information",
               "--output", str(OUTPUT)]
    write_json(OUTPUT / "launcher.json", {
        **identity, "command": command, "budget_seconds": BUDGET_SECONDS,
        "scope": "fixed_homogeneous_pretrained_contextual_state_and_numerical_assay",
        "expected_records": EXPECTED_RECORDS, "primary_records": 128,
        "model_calls_expected": 2048, "maximum_model_checkpoints": 1,
        "threads": {name: environment[name] for name in
                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "artifact_retention_days": 90,
    })
    outcome = run_child(command, OUTPUT / "worker.log", environment, BUDGET_SECONDS)
    retention_started = time.perf_counter()
    accounting = completion_accounting(OUTPUT)
    write_json(OUTPUT / "completion_accounting.json", accounting)
    compression, retention_errors = {}, []
    for path in sorted(OUTPUT.rglob("*.jsonl")):
        if path.is_symlink():
            retention_errors.append(
                {"file": path.relative_to(OUTPUT).as_posix(), "type": "UnexpectedSymlink"})
            continue
        try:
            compression[path.relative_to(OUTPUT).as_posix()] = preserve_gzip(path)
        except Exception as error:
            retention_errors.append({
                "file": path.relative_to(OUTPUT).as_posix(),
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
            "All fixed records and required validation must complete. A missed "
            "primary numerical target or absent exact collisions is a valid outcome."),
        "run_kind": "preregistered_contextual_information_v1",
        "artifact_retention_days": 90, "compression_receipts": compression,
        "retention_errors": retention_errors, "retained_files": inventory,
        "all_slots_recorded": accounting["all_slots_recorded"],
        "all_records_completed_successfully": accounting["all_records_completed_successfully"],
        "empirical_retries": 0, "parent_model_imports_or_inference": 0,
        "supervisor_retention_wall_seconds": time.perf_counter() - retention_started,
        "retention_budget_scope": (
            "Post-child byte-preserving packaging and missing-slot accounting; "
            "no model, analysis, or scientific work resumes after the 600-second deadline."),
    }
    write_json(OUTPUT / "execution.json", execution)
    for name in ("results.json", "report.md", "status.json", "runtime_manifest.json",
                 "model_manifest.json", "tokenization.json",
                 "worker_failure.json", "completion_accounting.json",
                 "worker.log", "execution.json"):
        emit_file(OUTPUT / name)
    print(json.dumps({
        "worker_returncode": outcome["worker_returncode"],
        "failure": outcome["failure"], "retention_errors": retention_errors,
        "retained_files": len(inventory),
        "child_wall_seconds": outcome["child_wall_seconds_including_imports"],
        "all_slots_recorded": accounting["all_slots_recorded"],
        "all_records_completed_successfully": accounting["all_records_completed_successfully"],
    }, sort_keys=True), flush=True)
    if (outcome["worker_returncode"] != 0 or outcome["failure"] or retention_errors
            or not accounting["all_records_completed_successfully"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
