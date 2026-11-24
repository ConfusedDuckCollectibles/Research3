"""Execute the preregistered binding study once, with a hard process budget.

The parent bounds an isolated process group, including downloads and all model
work. Completed and partial outputs survive timeout. This wrapper never changes
study settings, repeats fits, or resumes an interrupted empirical run.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time

from run_pretrained_pilot import DOWNLOAD_FILES, MODELS, file_digest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "checkpoints" / "binding-study"
DESIGN_COMMIT = "dfce06b5dfb2860414af166ef4441eefd9548afc"
HELPER_COMMIT = "9eb4478fae1ad373234eb2a725345b66c8c63c57"
BUDGET_SECONDS = 1800
PINNED_PACKAGES = {
    "torch": "2.6.0+cpu", "transformers": "4.48.2", "numpy": "2.4.6",
    "tokenizers": "0.21.4", "safetensors": "0.8.0", "huggingface-hub": "0.36.2",
}
HISTORICAL_HELPERS = (
    "latent_bridge/query_conditioned.py",
    "latent_bridge/query_conditioned_data.py",
    "latent_bridge/counterfactual_runner.py",
    "latent_bridge/private_counterfactual.py",
    "latent_bridge/transformers_runner.py",
    "latent_bridge/private_lookup.py",
    "scripts/run_pretrained_pilot.py",
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, timeout=20
    ).stdout


def source_manifest() -> Path:
    design_name = "docs/binding_design.md"
    design_expected = git_bytes("show", f"{DESIGN_COMMIT}:{design_name}")
    design_actual = (ROOT / design_name).read_bytes()
    historical = {}
    for name in HISTORICAL_HELPERS:
        expected = git_bytes("show", f"{HELPER_COMMIT}:{name}")
        actual = (ROOT / name).read_bytes()
        historical[name] = {
            "expected_sha256": hashlib.sha256(expected).hexdigest(),
            "actual_sha256": hashlib.sha256(actual).hexdigest(),
        }
    names = set(HISTORICAL_HELPERS) | {
        design_name, "scripts/run_binding_study.py",
        ".github/workflows/binding-study.yml",
        ".github/workflows/binding-validation.yml", "pyproject.toml",
    }
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / "latent_bridge").glob("binding_*.py"))
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_binding*.py"))
    if not {"latent_bridge/binding_data.py", "latent_bridge/binding_study.py"} <= names:
        raise RuntimeError("Required binding implementation absent")
    files = {name: file_digest(ROOT / name) for name in sorted(names)}
    all_match = design_expected == design_actual and all(
        item["expected_sha256"] == item["actual_sha256"] for item in historical.values()
    )
    commit = git_bytes("rev-parse", "HEAD").decode().strip()
    manifest = {
        "github_commit": commit, "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "frozen_design_commit": DESIGN_COMMIT,
        "design_sha256": hashlib.sha256(design_actual).hexdigest(),
        "expected_design_sha256": hashlib.sha256(design_expected).hexdigest(),
        "files": files, "historical_helper_commit": HELPER_COMMIT,
        "historical_helpers": historical, "all_match": all_match,
        "verification": "Expected design and historical helper bytes come from immutable git commits.",
    }
    path = OUTPUT / "code_manifest.json"
    write_json(path, manifest)
    if not all_match:
        raise RuntimeError("Frozen design or historical helpers changed; no model work started")
    if os.environ.get("GITHUB_SHA") and commit != os.environ["GITHUB_SHA"]:
        raise RuntimeError("Checked-out source differs from workflow commit")
    git_bytes("diff", "--exit-code", "HEAD", "--", *sorted(names))
    return path


def verify_runtime() -> None:
    actual = {name: importlib.metadata.version(name) for name in PINNED_PACKAGES}
    value = {
        "python": platform.python_version(), "expected_python": "3.11.18",
        "actual_packages": actual, "expected_packages": PINNED_PACKAGES,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "github_commit": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "torch_threads": 2,
    }
    write_json(OUTPUT / "runtime_manifest.json", value)
    if value["python"] != value["expected_python"] or actual != PINNED_PACKAGES:
        raise RuntimeError("Pinned runtime differs; no model work started")


def worker() -> None:
    phases = {}
    current = None
    phase_started = None

    def begin(name: str) -> None:
        nonlocal current, phase_started
        current, phase_started = name, time.perf_counter()
        write_json(OUTPUT / "worker_phases.json", {"current": current, "completed_seconds": phases})
        print(f"Starting {name}", flush=True)

    def finish() -> None:
        nonlocal current, phase_started
        phases[current] = time.perf_counter() - phase_started
        current, phase_started = None, None
        write_json(OUTPUT / "worker_phases.json", {"current": None, "completed_seconds": phases})

    try:
        begin("runtime_and_source_preflight")
        verify_runtime()
        code_path = source_manifest()
        code = json.loads(code_path.read_text(encoding="utf-8"))
        finish()

        begin("download_pinned_public_models")
        from huggingface_hub import snapshot_download
        model_manifest = {
            "status": "downloading_pinned_public_models", "models": {},
            "python": platform.python_version(), "platform": platform.platform(),
            "github_commit": code["github_commit"], "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "design_sha256": code["design_sha256"],
            "remote_code_allowed": False, "hub_credentials_used": False,
        }
        model_path = OUTPUT / "model_manifest.json"
        write_json(model_path, model_manifest)
        for role, spec in MODELS.items():
            destination = ROOT / "local_models" / ("binding_" + role)
            before = time.perf_counter()
            print(f"Downloading {role}: {spec['id']} @ {spec['revision']}", flush=True)
            snapshot_download(
                repo_id=spec["id"], revision=spec["revision"], local_dir=str(destination),
                allow_patterns=DOWNLOAD_FILES, token=False, max_workers=2,
            )
            if not (destination / "model.safetensors").is_file():
                raise RuntimeError(f"Pinned {role} snapshot lacks safetensors weights")
            model_manifest["models"][role] = {
                **spec,
                "files": {p.name: file_digest(p) for p in sorted(destination.iterdir()) if p.is_file()},
                "download_seconds": time.perf_counter() - before,
            }
            write_json(model_path, model_manifest)
        model_manifest["status"] = "downloaded_pinned_public_pretrained_models"
        model_manifest["download_seconds"] = sum(v["download_seconds"] for v in model_manifest["models"].values())
        write_json(model_path, model_manifest)
        finish()

        environment = dict(os.environ)
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        data = OUTPUT / "data"
        commands = [
            ("generate_frozen_data", [
                sys.executable, "-m", "latent_bridge.binding_data", "--output", str(data),
            ]),
            ("fit_gate_and_evaluate", [
                sys.executable, "-m", "latent_bridge.binding_study",
                "--sender-model", str(ROOT / "local_models" / "binding_sender"),
                "--receiver-model", str(ROOT / "local_models" / "binding_receiver"),
                "--data-dir", str(data), "--output", str(OUTPUT),
                "--model-manifest", str(model_path), "--code-manifest", str(code_path),
                "--threads", "2",
            ]),
        ]
        write_json(OUTPUT / "commands.json", {"commands": [{"phase": k, "argv": v} for k, v in commands]})
        for name, arguments in commands:
            begin(name)
            # Descendants inherit this worker's isolated process group. The parent
            # kills the entire group when the single global deadline expires.
            subprocess.run(arguments, cwd=ROOT, env=environment, check=True)
            finish()
    except BaseException as exc:
        write_json(OUTPUT / "worker_failure.json", {
            "type": type(exc).__name__, "message": str(exc), "phase": current,
            "phase_elapsed_seconds": None if phase_started is None else time.perf_counter() - phase_started,
            "completed_phase_seconds": phases,
        })
        raise


def require_first_attempt() -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("This frozen study must execute through its recorded GitHub CPU workflow")
    if os.environ.get("GITHUB_RUN_ATTEMPT") != "1" or os.environ.get("GITHUB_RUN_NUMBER") != "1":
        raise RuntimeError("Empirical reruns are disabled: only workflow run 1, attempt 1 is authorized")
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise RuntimeError("Frozen study must execute from main")
    if (ROOT / "results" / "binding_study").exists():
        raise RuntimeError("Published binding outcomes already exist; this study cannot be repeated")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError("Output already contains work; an interrupted study cannot be resumed")


def compress_partial_predictions() -> dict:
    receipts = {}
    for name in ("predictions.jsonl", "validation_predictions.jsonl"):
        path = OUTPUT / name
        if not path.is_file():
            continue
        compressed = path.with_name(path.name + ".gz")
        # The runner may already have recorded this gzip's exact file hash.
        # Preserve those bytes: gzip.compress and GzipFile can emit different
        # headers on the pinned Python version despite identical raw content.
        if not compressed.exists():
            with path.open("rb") as source, compressed.open("wb") as target:
                with gzip.GzipFile(filename="", fileobj=target, mode="wb", compresslevel=9, mtime=0) as z:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        z.write(chunk)
        digest = hashlib.sha256()
        with gzip.open(compressed, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        raw = file_digest(path)
        if digest.hexdigest() != raw["sha256"]:
            # Never conceal a truncated/mismatched previously recorded file.
            # Raw and compressed bytes both remain available for inspection.
            raise RuntimeError("Existing compressed prediction bytes differ from raw; both preserved")
        receipts[name] = {"raw": raw, "gzip": file_digest(compressed), "round_trip_verified": True}
    return receipts


def emit_small_reports() -> None:
    for name in (
        "status.json", "competence_gate.json", "results.json", "report.md",
        "runtime_manifest.json", "code_manifest.json", "model_manifest.json",
        "worker_failure.json", "execution.json",
    ):
        path = OUTPUT / name
        if path.is_file() and path.stat().st_size <= 300_000:
            payload = path.read_text(encoding="utf-8")
            print(f"BINDING_FILE_BEGIN {name} {file_digest(path)['sha256']}", flush=True)
            print(payload, end="" if payload.endswith("\n") else "\n", flush=True)
            print(f"BINDING_FILE_END {name}", flush=True)


def main() -> None:
    started = time.perf_counter()
    require_first_attempt()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    failure, returncode, process, pump = None, None, None, None
    log = None
    try:
        environment = dict(os.environ)
        environment.update(
            PYTHONUNBUFFERED="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
            HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
        )
        command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
        write_json(OUTPUT / "launch.json", {
            "command": command, "budget_seconds": BUDGET_SECONDS,
            "github_commit": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "github_run_number": os.environ.get("GITHUB_RUN_NUMBER"),
            "frozen_design_commit": DESIGN_COMMIT, "threads": 2,
            "time_budget_scope": "All worker imports, downloads, data preparation, fits and fresh scoring.",
        })
        log = (OUTPUT / "worker.log").open("wb")
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        def copy_output() -> None:
            for chunk in iter(process.stdout.readline, b""):
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()

        pump = threading.Thread(target=copy_output, daemon=True)
        pump.start()
        remaining = BUDGET_SECONDS - (time.perf_counter() - started)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, BUDGET_SECONDS)
        returncode = process.wait(timeout=remaining)
        if returncode:
            failure = {"type": "WorkerExit", "returncode": returncode}
    except subprocess.TimeoutExpired:
        failure = {"type": "GlobalStudyTimeout", "budget_seconds": BUDGET_SECONDS}
        if process is not None:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=10)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=10)
    finally:
        study_seconds = time.perf_counter() - started
        if pump is not None:
            pump.join(timeout=10)
        if log is not None:
            log.close()
        prediction_receipts = {}
        try:
            prediction_receipts = compress_partial_predictions()
        except Exception as exc:
            failure = failure or {"type": type(exc).__name__, "message": str(exc)}
        retained = {
            str(p.relative_to(OUTPUT)): file_digest(p)
            for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name != "execution.json"
        }
        study_status = None
        if (OUTPUT / "status.json").is_file():
            try:
                study_status = json.loads((OUTPUT / "status.json").read_text(encoding="utf-8"))
            except (ValueError, OSError):
                study_status = {"status": "unreadable_partial_status"}
        execution = {
            "status": "worker_finished" if failure is None else "stopped_with_recorded_failure",
            "study_status": study_status, "failure": failure, "worker_returncode": returncode,
            "run_kind": "preregistered_binding_auxiliary_loss_study",
            "github_commit": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "github_run_number": os.environ.get("GITHUB_RUN_NUMBER"),
            "frozen_design_commit": DESIGN_COMMIT,
            "design_sha256": hashlib.sha256((ROOT / "docs/binding_design.md").read_bytes()).hexdigest(),
            "budget_seconds": BUDGET_SECONDS, "study_wall_seconds_including_downloads": study_seconds,
            "wrapper_wall_seconds_including_retention": time.perf_counter() - started,
            "artifact_retention_days": 90, "prediction_receipts": prediction_receipts,
            "retained_files": retained,
            "completion_rule": "Interpret competence_gate/status/results together; exit zero alone is not a complete-study claim.",
        }
        write_json(OUTPUT / "execution.json", execution)
        emit_small_reports()
        print(json.dumps({
            "status": execution["status"], "failure": failure,
            "study_wall_seconds": study_seconds, "retained_file_count": len(retained),
        }, sort_keys=True), flush=True)
    if failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker()
    elif sys.argv[1:]:
        raise SystemExit("No study overrides or resume arguments are accepted")
    else:
        main()
