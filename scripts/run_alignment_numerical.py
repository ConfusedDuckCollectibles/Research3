"""Bound one frozen, CPU-only numerical study and preserve partial evidence."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "checkpoints/alignment-numerical"
BUDGET_SECONDS = 300
DESIGN_COMMIT = "e7c467f2b097666dc474a3dbb6e273bf253e785f"
DESIGN_PATH = "docs/alignment_numerical_design.md"
DESIGN_SHA256 = "c785e3fef3af9c8fac3bb7a04ccd7980ddfa51c121f3f11a9068eda3b7a70a5e"
THEORY_SHA256 = "06fa7747aa96db8719ce7ae930789daee6291c65cab6752fcadb1364f0f6f2ff"
SOURCE_PATHS = (
    DESIGN_PATH, "docs/alignment_geometry.md",
    "latent_bridge/__init__.py", "latent_bridge/alignment_core.py",
    "latent_bridge/alignment_packet.py", "latent_bridge/alignment_numerical.py",
    "scripts/run_alignment_numerical.py", "tests/test_alignment_execution.py",
    "tests/test_alignment_packet.py", "tests/test_alignment_core.py",
    ".github/workflows/alignment-numerical-v1.yml", "pyproject.toml",
)


def info(path: Path) -> dict:
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": count}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def require_context() -> dict:
    env = os.environ
    if (env.get("GITHUB_ACTIONS") != "true"
            or env.get("GITHUB_REPOSITORY") != "ConfusedDuckCollectibles/Research3"
            or env.get("GITHUB_REF") != "refs/heads/main"
            or env.get("GITHUB_RUN_ATTEMPT") != "1"
            or env.get("GITHUB_RUN_NUMBER") != "1"):
        raise RuntimeError("This wrapper permits only the first original v1 main workflow run")
    if platform.python_version() != "3.11.18":
        raise RuntimeError("Frozen Python runtime mismatch")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()
    if head != env.get("GITHUB_SHA"):
        raise RuntimeError("Checkout and workflow source differ")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError("Refusing to replace an existing numerical output directory")
    if (ROOT / "results/alignment_numerical").exists():
        raise RuntimeError("Original numerical results already exist; do not repeat them")
    if info(ROOT / DESIGN_PATH)["sha256"] != DESIGN_SHA256:
        raise RuntimeError("Frozen numerical design changed")
    if info(ROOT / "docs/alignment_geometry.md")["sha256"] != THEORY_SHA256:
        raise RuntimeError("Reviewed mathematical note changed")
    return {"github_commit": head, "github_run_id": env["GITHUB_RUN_ID"],
            "github_run_number": 1, "github_run_attempt": 1,
            "frozen_design_commit": DESIGN_COMMIT, "design_sha256": DESIGN_SHA256}


def run_child(command: list[str], log_path: Path, environment: dict, budget: float) -> dict:
    """Kill the entire new process group at the deadline; never resume a prefix."""
    process = None
    started = time.perf_counter()
    failure = None
    returncode = None
    with log_path.open("wb") as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       start_new_session=True, stdout=log,
                                       stderr=subprocess.STDOUT)
            remaining = budget - (time.perf_counter() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, budget)
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            failure = {"type": "NumericalStudyTimeout", "budget_seconds": budget}
        except BaseException as error:
            failure = {"type": type(error).__name__, "message": str(error)}
        finally:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = process.wait(timeout=10)
    return {"worker_returncode": returncode, "failure": failure,
            "child_wall_seconds_including_imports": time.perf_counter() - started}


def preserve_gzip(path: Path) -> dict:
    """Preserve existing gzip bytes; create only an absent equivalent."""
    original = info(path)
    target = path.with_name(path.name + ".gz")
    if not target.exists():
        temporary = target.with_name(target.name + ".tmp")
        with path.open("rb") as source, temporary.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as encoded:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    encoded.write(chunk)
        temporary.replace(target)
    digest, count = hashlib.sha256(), 0
    with gzip.open(target, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
            if count > original["bytes"]:
                raise RuntimeError("Compressed numerical evidence exceeds original size")
    if {"sha256": digest.hexdigest(), "bytes": count} != original:
        raise RuntimeError("Compressed numerical evidence does not reproduce original bytes")
    return {"raw": original, "gzip": info(target), "round_trip_verified": True}


def emit_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size > 300_000:
        return
    raw = path.read_bytes()
    raw.decode("utf-8")
    print("ALIGNMENT_FILE_BEGIN " + path.name + " " + hashlib.sha256(raw).hexdigest()
          + " " + str(len(raw)), flush=True)
    sys.stdout.buffer.write(raw)
    if not raw.endswith(b"\n"):
        sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()
    print("ALIGNMENT_FILE_END " + path.name, flush=True)


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
    command = [sys.executable, "-m", "latent_bridge.alignment_numerical",
               "--output", str(OUTPUT)]
    write_json(OUTPUT / "launcher.json",
               {**identity, "command": command, "budget_seconds": BUDGET_SECONDS,
                "scope": "fixed_numerical_study_no_models_or_network",
                "threads": {name: environment[name] for name in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "artifact_retention_days": 90})
    outcome = run_child(command, OUTPUT / "worker.log", environment, BUDGET_SECONDS)
    compression = {}
    retention_errors = []
    for name in ("cases.jsonl", "timings.jsonl"):
        path = OUTPUT / name
        if path.exists():
            try:
                compression[name] = preserve_gzip(path)
            except Exception as error:
                retention_errors.append({"file": name, "type": type(error).__name__,
                                         "message": str(error)})
    inventory = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_symlink():
            retention_errors.append({"file": str(path.relative_to(OUTPUT)),
                                     "type": "UnexpectedSymlink"})
        elif path.is_file() and path.name != "execution.json":
            inventory[path.relative_to(OUTPUT).as_posix()] = info(path)
    execution = {**identity, **outcome, "budget_seconds": BUDGET_SECONDS,
                 "completion_rule": "A zero exit is necessary, but the retained required-case checks determine scientific verification.",
                 "run_kind": "preregistered_alignment_numerical_study_v1",
                 "artifact_retention_days": 90, "compression_receipts": compression,
                 "retention_errors": retention_errors, "retained_files": inventory,
                 "benchmark_retries": 0, "model_imports_or_inference": 0}
    write_json(OUTPUT / "execution.json", execution)
    for name in ("results.json", "report.md", "status.json", "runtime_manifest.json",
                 "worker_failure.json", "worker.log", "execution.json"):
        emit_file(OUTPUT / name)
    print(json.dumps({"worker_returncode": outcome["worker_returncode"],
                      "failure": outcome["failure"], "retention_errors": retention_errors,
                      "retained_files": len(inventory),
                      "child_wall_seconds": outcome["child_wall_seconds_including_imports"]},
                     sort_keys=True))
    if outcome["worker_returncode"] != 0 or outcome["failure"] or retention_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
